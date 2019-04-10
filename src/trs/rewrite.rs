//! Polymorphically typed (Hindley-Milner) First-Order Term Rewriting Systems (no abstraction)
//!
//! Much thanks to:
//! - https://github.com/rob-smallshire/hindley-milner-python
//! - https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system
//! - (TAPL; Pierce, 2002, ch. 22)

use itertools::Itertools;
use polytype::Context as TypeContext;
use rand::seq::sample_iter;
use rand::Rng;
use std::f64::NEG_INFINITY;
use std::fmt;
use term_rewriting::trace::Trace;
use term_rewriting::{Rule, RuleContext, Strategy as RewriteStrategy, Term, TRS as UntypedTRS};

use super::{Lexicon, ModelParams, SampleError, TypeError};

/// Manages the semantics of a term rewriting system.
#[derive(Debug, PartialEq, Clone)]
pub struct TRS {
    pub(crate) lex: Lexicon,
    // INVARIANT: UntypedTRS.rules ends with lex.background
    pub(crate) utrs: UntypedTRS,
    pub(crate) ctx: TypeContext,
}
impl TRS {
    /// Create a new `TRS` under the given [`Lexicon`]. Any background knowledge
    /// will be appended to the given ruleset.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate term_rewriting;
    /// # use programinduction::trs::{TRS, Lexicon};
    /// # use term_rewriting::{Signature, parse_rule};
    /// # use polytype::Context as TypeContext;
    /// # fn main() {
    /// let mut sig = Signature::default();
    ///
    /// let mut ops = vec![];
    /// sig.new_op(2, Some("PLUS".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int), tp!(int)]]);
    /// sig.new_op(1, Some("SUCC".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int)]]);
    /// sig.new_op(0, Some("ZERO".to_string()));
    /// ops.push(ptp![int]);
    ///
    /// let rules = vec![
    ///     parse_rule(&mut sig, "PLUS(x_ ZERO) = x_").expect("parsed rule"),
    ///     parse_rule(&mut sig, "PLUS(x_ SUCC(y_)) = SUCC(PLUS(x_ y_))").expect("parsed rule"),
    /// ];
    ///
    /// let vars = vec![
    ///     ptp![int],
    ///     ptp![int],
    ///     ptp![int],
    /// ];
    ///
    /// let lexicon = Lexicon::from_signature(sig, ops, vars, vec![], vec![], false, TypeContext::default());
    ///
    /// let ctx = lexicon.context();
    ///
    /// let trs = TRS::new(&lexicon, rules, &ctx).unwrap();
    ///
    /// assert_eq!(trs.size(), 12);
    /// # }
    /// ```
    /// [`Lexicon`]: struct.Lexicon.html
    pub fn new(
        lexicon: &Lexicon,
        mut rules: Vec<Rule>,
        ctx: &TypeContext,
    ) -> Result<TRS, TypeError> {
        let lexicon = lexicon.clone();
        let mut ctx = ctx.clone();
        let utrs = {
            let lex = lexicon.0.read().expect("poisoned lexicon");
            rules.append(&mut lex.background.clone());
            let utrs = UntypedTRS::new(rules);
            lex.infer_utrs(&utrs, &mut ctx)?;
            utrs
        };
        Ok(TRS {
            lex: lexicon,
            utrs,
            ctx,
        })
    }

    /// The size of the underlying [`term_rewriting::TRS`].
    ///
    /// [`term_rewriting::TRS`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.TRS.html#method.size
    pub fn size(&self) -> usize {
        self.utrs.size()
    }

    /// The length of the underlying [`term_rewriting::TRS`].
    ///
    /// [`term_rewriting::TRS`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.TRS.html#method.size
    pub fn len(&self) -> usize {
        self.utrs.len()
    }

    /// Is the underlying [`term_rewriting::TRS`] empty?.
    ///
    /// [`term_rewriting::TRS`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.TRS.html#method.size
    pub fn is_empty(&self) -> bool {
        self.utrs.is_empty()
    }

    /// A pseudo log prior for a `TRS`: the negative [`size`] of the `TRS`.
    ///
    /// [`size`]: struct.TRS.html#method.size
    pub fn pseudo_log_prior(&self) -> f64 {
        -(self.size() as f64)
    }

    /// A log likelihood for a `TRS`: the probability of `data`'s RHSs appearing
    /// in [`term_rewriting::Trace`]s rooted at its LHSs.
    ///
    /// [`term_rewriting::Trace`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/trace/struct.Trace.html
    pub fn log_likelihood(&self, data: &[Rule], params: ModelParams) -> f64 {
        data.iter()
            .map(|x| self.single_log_likelihood(x, params))
            .sum()
    }

    /// test the likelihood
    pub fn test_single_likelihood(&self, datum: &Rule, params: ModelParams) {
        println!("datum: {}", datum.pretty());
        let ll = if let Some(ref rhs) = datum.rhs() {
            let mut trace = Trace::new(
                &self.utrs,
                &datum.lhs,
                params.p_observe,
                params.max_size,
                RewriteStrategy::All,
            );
            for i in 0..params.max_steps {
                let leaf = trace.next().unwrap();
                println!(
                    "{}, {:?}, {}, {}, {}",
                    i,
                    leaf.state(),
                    leaf.term().pretty(),
                    leaf.log_p(),
                    leaf.depth()
                );
            }
            println!("trace size: {}", trace.size());
            let partial = trace.rewrites_to(params.max_steps, rhs);
            println!("trace size: {}", trace.size());
            partial
        } else {
            println!("ouch! NEG_INFINITY");
            NEG_INFINITY
        };
        println!("partial result: {}", ll);

        let result = if ll == NEG_INFINITY {
            params.p_partial.ln()
        } else {
            (1.0 - params.p_partial).ln() + ll
        };
        println!("computed likelihood: {}", result);
    }

    /// Compute the log likelihood for a single datum.
    fn single_log_likelihood(&self, datum: &Rule, params: ModelParams) -> f64 {
        let ll = if let Some(ref rhs) = datum.rhs() {
            let mut trace = Trace::new(
                &self.utrs,
                &datum.lhs,
                params.p_observe,
                params.max_size,
                RewriteStrategy::All,
            );
            trace.rewrites_to(params.max_steps, rhs)
        } else {
            NEG_INFINITY
        };

        if ll == NEG_INFINITY {
            params.p_partial.ln()
        } else {
            (1.0 - params.p_partial).ln() + ll
        }
    }

    /// Combine [`pseudo_log_prior`] and [`log_likelihood`], failing early if the
    /// prior is `0.0`.
    ///
    /// [`pseudo_log_prior`]: struct.TRS.html#method.pseudo_log_prior
    /// [`log_likelihood`]: struct.TRS.html#method.log_likelihood
    pub fn posterior(&self, data: &[Rule], params: ModelParams) -> f64 {
        let prior = self.pseudo_log_prior();
        if prior == NEG_INFINITY {
            NEG_INFINITY
        } else {
            prior + self.log_likelihood(data, params)
        }
    }

    /// Sample a rule and add it to the rewrite system.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate rand;
    /// # extern crate term_rewriting;
    /// # use programinduction::trs::{TRS, Lexicon};
    /// # use polytype::Context as TypeContext;
    /// # use rand::{thread_rng};
    /// # use term_rewriting::{Context, RuleContext, Signature, parse_rule};
    /// # fn main() {
    /// let mut sig = Signature::default();
    ///
    /// let mut ops = vec![];
    /// sig.new_op(2, Some(".".to_string()));
    /// ops.push(ptp![0, 1; @arrow[tp!(@arrow[tp!(0), tp!(1)]), tp!(0), tp!(1)]]);
    /// sig.new_op(2, Some("PLUS".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int), tp!(int)]]);
    /// sig.new_op(1, Some("SUCC".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int)]]);
    /// sig.new_op(0, Some("ZERO".to_string()));
    /// ops.push(ptp![int]);
    ///
    /// let rules = vec![
    ///     parse_rule(&mut sig, "PLUS(x_ ZERO) = x_").expect("parsed rule"),
    ///     parse_rule(&mut sig, "PLUS(x_ SUCC(y_)) = SUCC(PLUS(x_ y_))").expect("parsed rule"),
    /// ];
    ///
    /// let vars = vec![
    ///     ptp![int],
    ///     ptp![int],
    ///     ptp![int],
    /// ];
    ///
    /// for op in sig.operators() {
    ///     println!("{:?}/{}", op.name(), op.arity())
    /// }
    /// for r in &rules {
    ///     println!("{:?}", r.pretty());
    /// }
    /// let lexicon = Lexicon::from_signature(sig, ops, vars, vec![], vec![], false, TypeContext::default());
    ///
    /// let mut trs = TRS::new(&lexicon, rules, &lexicon.context()).unwrap();
    ///
    /// assert_eq!(trs.len(), 2);
    ///
    /// let contexts = vec![
    ///     RuleContext {
    ///         lhs: Context::Hole,
    ///         rhs: vec![Context::Hole],
    ///     }
    /// ];
    /// let mut rng = thread_rng();
    /// let atom_weights = (0.5, 0.25, 0.25);
    /// let max_size = 50;
    ///
    /// if let Ok(new_trs) = trs.add_rule(&contexts, atom_weights, max_size, &mut rng) {
    ///     assert_eq!(new_trs.len(), 3);
    /// } else {
    ///     assert_eq!(trs.len(), 2);
    /// }
    /// # }
    /// ```
    pub fn add_rule<R: Rng>(
        &self,
        contexts: &[RuleContext],
        atom_weights: (f64, f64, f64),
        max_size: usize,
        rng: &mut R,
    ) -> Result<TRS, SampleError> {
        let mut trs = self.clone();
        let context = sample_iter(rng, contexts, 1)?[0].clone();
        let rule =
            trs.lex
                .sample_rule_from_context(context, &mut trs.ctx, atom_weights, true, max_size)?;
        trs.lex
            .0
            .write()
            .expect("poisoned lexicon")
            .infer_rule(&rule, &mut trs.ctx)?;
        trs.utrs.push(rule)?;
        Ok(trs)
    }
    /// Delete a rule from the rewrite system if possible. Background knowledge
    /// cannot be deleted.
    pub fn delete_rule<R: Rng>(&self, rng: &mut R) -> Result<TRS, SampleError> {
        let background = &self.lex.0.read().expect("poisoned lexicon").background;
        let clauses = self.utrs.clauses();
        let deletable: Vec<_> = clauses.iter().filter(|c| !background.contains(c)).collect();
        if deletable.is_empty() {
            Err(SampleError::OptionsExhausted)
        } else {
            let mut trs = self.clone();
            trs.utrs.remove_clauses(sample_iter(rng, deletable, 1)?[0])?;
            Ok(trs)
        }
    }
    /// Replaces one subterm with another subterm in a main Term.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate rand;
    /// # extern crate term_rewriting;
    /// # use programinduction::trs::{TRS, Lexicon};
    /// # use rand::{thread_rng};
    /// # use term_rewriting::{Context, RuleContext, Signature, parse_rule, parse_term};
    /// # fn main() {
    /// let mut sig = Signature::default();
    /// let term = parse_term(&mut sig, "A(y_ B(C) D)").expect("parse of A(y_ B(C) D)");
    /// let t = parse_term(&mut sig, "B(C)").expect("parse of B(C)");
    /// let v = parse_term(&mut sig, "x_").expect("parse of x_");
    ///
    /// let new_term = TRS::replace_term_helper(&term, &t, v);
    ///
    /// assert_eq!(new_term.display(), "A(y_ x_ D)");
    /// # }
    /// ```
    pub fn replace_term_helper(term: &Term, t: &Term, v: Term) -> Term {
        if Term::alpha(t, term) != None {
            let unwrapped_sub = Term::alpha(t, term).unwrap();
            return v.substitute(&unwrapped_sub);
        } else if term.args() != vec![] {
            match *term {
                Term::Variable(ref _var) => {
                    return term.clone();
                }
                Term::Application { ref op, args: _ } => {
                    let mut args = term.args().clone();
                    for idx in 0..args.len() {
                        args[idx] = TRS::replace_term_helper(&args[idx], t, v.clone());
                    }
                    let op = op.clone();
                    return Term::Application { op, args };
                }
            }
        }
        return term.clone();
    }
    /// Replaces one subterm with another subterm in a given Rule.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate rand;
    /// # extern crate term_rewriting;
    /// # use programinduction::trs::{TRS, Lexicon};
    /// # use rand::{thread_rng};
    /// # use term_rewriting::{Context, RuleContext, Signature, parse_rule, parse_term};
    /// # fn main() {
    /// let mut sig = Signature::default();
    /// let rule = parse_rule(&mut sig, "A(y_ B(C) D) = B(C)").expect("parse of A(y_ B(C) D) = B(C)");
    /// let t = parse_term(&mut sig, "B(C)").expect("parse of B(C)");
    /// let v = parse_term(&mut sig, "x_").expect("parse of x_");
    ///
    /// let new_rule = TRS::replace_term_in_rule_helper(&rule, &t, v);
    /// if new_rule == None {
    ///     assert!(false);
    /// } else {
    ///     let rule = new_rule.unwrap();
    ///     assert_eq!(rule.display(), "A(y_ x_ D) = x_");
    /// }
    /// # }
    /// ```
    pub fn replace_term_in_rule_helper(rule: &Rule, t: &Term, v: Term) -> Option<Rule> {
        let r = rule.clone();
        let lhs = TRS::replace_term_helper(&r.lhs, t, v.clone());
        let mut rhs: Vec<Term> = vec![];
        for idx in 0..r.rhs.len() {
            rhs.push(TRS::replace_term_helper(&r.rhs[idx].clone(), t, v.clone()));
        }
        Rule::new(lhs, rhs)
    }
    // helper for routinization
    // TODO fix rule lhs before inserting
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate rand;
    /// # extern crate term_rewriting;
    /// # use programinduction::trs::{TRS, Lexicon};
    /// # use rand::{thread_rng};
    /// # use term_rewriting::{Context, RuleContext, Signature, parse_rule, parse_term};
    /// # fn main() {
    /// let mut sig = Signature::default();
    ///
    /// let t = parse_term(&mut sig, "A(B(x_ x_))").expect("parse of A(B(x_ x_))");
    /// let r = parse_rule(&mut sig, "C(y_) = B(y_ y_)").expect("parse of C(y_) = B(y_ y_)");
    /// let mut rng = thread_rng();
    ///
    /// let result = TRS::inverse_evaluation_helper(&r, &t, &mut rng);
    ///
    /// assert_eq!(result.display(), "A(C(x_))");
    /// # }
    /// ```
    pub fn inverse_evaluation_helper<R: Rng>(rule: &Rule, t: &Term, rng: &mut R) -> Term {
        let rhs_idx = rng.gen_range(0, rule.rhs.len());
        TRS::replace_term_helper(t, &rule.rhs[rhs_idx], rule.lhs.clone())
    }
    /// Given two rules, attempts to apply one rule inversely to the other.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate rand;
    /// # extern crate term_rewriting;
    /// # use programinduction::trs::{TRS, Lexicon};
    /// # use rand::{thread_rng};
    /// # use term_rewriting::{Context, RuleContext, Signature, parse_rule, parse_term};
    /// # fn main() {
    /// let mut sig = Signature::default();
    ///
    /// let r = parse_rule(&mut sig, "A(B(x_ x_)) = D B(x_ x_)").expect("parse of A(B(x_ x_)) = D B(x_ x_)");
    /// let rule = parse_rule(&mut sig, "C(y_) = B(y_ y_)").expect("parse of C(y_) = B(y_ y_)");
    /// let mut rng = thread_rng();
    ///
    /// let result = TRS::inverse_evaluate_rule_helper(&rule, &r, &mut rng);
    ///
    /// assert_eq!(result.unwrap().pretty(), "A(C(x_)) = D C(x_)");
    /// # }
    /// ```
    pub fn inverse_evaluate_rule_helper<R: Rng>(
        rule: &Rule,
        r: &Rule,
        rng: &mut R,
    ) -> Option<Rule> {
        let lhs = TRS::inverse_evaluation_helper(rule, &r.lhs, rng);
        let mut rhs: Vec<Term> = vec![];
        for idx in 0..r.rhs.len() {
            rhs.push(TRS::inverse_evaluation_helper(rule, &r.rhs[idx], rng));
        }
        Rule::new(lhs, rhs)
    }
    /// Selects two Rules from the TRS at random and atempts to inverse evaluate one rule on the other, if it
    /// succeeds it takes that new rule and inserts it imediately after the background.
    pub fn inverse_evaluate<R: Rng>(&self, rng: &mut R) -> Result<TRS, SampleError> {
        let mut trs = self.clone();
        let num_rules = self.len();
        let num_background = self
            .lex
            .0
            .read()
            .expect("poisoned lexicon")
            .background
            .len();
        if num_background >= num_rules - 1 {
            return Ok(trs);
        }
        let ref_idx: usize = rng.gen_range(0, num_rules);
        let mut target_idx: usize = rng.gen_range(num_background, num_rules);
        while ref_idx == target_idx {
            target_idx = rng.gen_range(num_background, num_rules);
        }
        let mut ref_rule = trs.utrs.rules[ref_idx].clone();
        let rhs_idx = rng.gen_range(0, ref_rule.rhs.len());
        let temp_rule = Rule::new(ref_rule.lhs, vec![ref_rule.rhs[rhs_idx].clone()]);
        if temp_rule == None {
            return Ok(trs);
        }
        ref_rule = temp_rule.unwrap();
        if ref_rule.lhs.variables().len() != ref_rule.rhs[0].variables().len() {
            return Ok(trs);
        }
        let new_rule =
            TRS::inverse_evaluate_rule_helper(&ref_rule, &trs.utrs.rules[target_idx], rng);
        if new_rule == None {
            return Ok(trs);
        }
        trs.utrs.remove_idx(target_idx).expect("removing old rule");
        trs.utrs
            .insert_idx(target_idx, new_rule.unwrap())
            .expect("inserting new rule");
        Ok(trs)
    }
}
impl fmt::Display for TRS {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let true_len = self.utrs.len()
            - self
                .lex
                .0
                .read()
                .expect("poisoned lexicon")
                .background
                .len();
        let trs_str = self
            .utrs
            .rules
            .iter()
            .take(true_len)
            .map(|r| format!("{};", r.display()))
            .join("\n");

        write!(f, "{}", trs_str)
    }
}
