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

use term_rewriting::{Rule, RuleContext, Strategy as RewriteStrategy, TRS as UntypedTRS};

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
    pub fn pretty_utrs(&self, sig: &Signature) -> String {
        self.utrs.pretty(sig)
    }
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
        let rule = trs.lex.sample_rule_from_context(
            context,
            &mut trs.ctx,
            atom_weights,
            true,
            max_size,
        )?;
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
            trs.utrs
                .remove_clauses(sample_iter(rng, deletable, 1)?[0])?;
            Ok(trs)
        }
    }
    pub fn add_exception<R: Rng>(&self, data: Vec<Rule>, rng: &mut R) -> Result<TRS, SampleError> {
        let num_background = self.lex.0.read().expect("poisoned lexicon").background.len();
        let mut trs = self.clone();
        let idx = rng.gen_range(0, data.len());
        trs.utrs.insert_idx(num_background, data[idx].clone())?;
        Ok(trs)
    }
    /// Move a rule from one place in the TRS to another.
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
    /// println!("{:?}", sig.operators());
    /// for op in sig.operators() {
    ///     println!("{:?}/{}", op.name(&sig), op.arity(&sig))
    /// }
    /// for r in &rules {
    ///     println!("{:?}", r);
    /// }
    /// let lexicon = Lexicon::from_signature(sig.clone(), ops, vars, vec![], false);
    ///
    /// let mut trs = TRS::new(&lexicon, rules).unwrap();
    ///
    /// let pretty_before = trs.pretty_utrs(&sig);
    ///
    /// let mut rng = thread_rng();
    ///
    /// let new_trs = trs.randomly_move_rule(&mut rng).expect("failed when moving rule");
    ///
    /// assert_ne!(pretty_before, new_trs.pretty_utrs(&sig));
    /// assert_eq!(pretty_before, "PLUS(x_, 0) = x_;\nPLUS(x_, SUCC(y_)) = SUCC(PLUS(x_, y_));");
    /// assert_eq!(new_trs.pretty_utrs(&sig), "PLUS(x_, SUCC(y_)) = SUCC(PLUS(x_, y_));\nPLUS(x_, 0) = x_;");
    /// # }
    /// ```
    pub fn randomly_move_rule<R: Rng>(&self, rng: &mut R) -> Result<TRS, SampleError> {
        let mut trs = self.clone();
        let num_rules = self.len();
        let num_background = self.lex.0.read().expect("poisoned lexicon").background.len();
        if num_background >= num_rules - 1 {
            return Ok(trs);
        }
        let i: usize = rng.gen_range(num_background, num_rules);
        let mut j: usize = rng.gen_range(num_background, num_rules);
        while j == i {
            j = rng.gen_range(0, num_rules);
        }
        trs.utrs
            .move_rule(i, j)
            .expect("moving rule from random locations i to j");
        Ok(trs)
    }
    /// replace helper
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
    /// assert_eq!(new_term.display(&sig), "A(y_ x_ D)");
    /// # }
    /// ```
    pub fn replace_term_helper(term: &Term, t: &Term, v: Term) -> Term {
        if Term::alpha(t, term) != None {
            return v;
        } else if term.args() != vec![] {
            match *term {
                Term::Variable(_var) => {
                    return term.clone();
                }
                Term::Application { op, args: _ } => {
                    let mut args = term.args().clone();
                    for idx in 0..args.len() {
                        args[idx] = TRS::replace_term_helper(&args[idx], t, v.clone());
                    }
                    return Term::Application { op, args };
                }
            }
        }
        return term.clone();
    }
    /// replaces term in rule with another
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
    ///     assert_eq!(rule.display(&sig), "A(y_ x_ D) = x_");
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
    /// swap lhs and rhs by randomly chosing one
    pub fn swap_lhs_and_r_rhs_helper<R: Rng>(rule: &Rule, rng: &mut R) -> Option<Rule> {
        let r = TRS::swap_lhs_and_all_rhs_helper(rule);
        if r == None {
            return None;
        }
        let rules = r.unwrap();
        let idx = rng.gen_range(0, rules.len());
        Some(rules[idx].clone())
    }
    /// swap lhs and rhs only if there is one
    /// returns none if they can not be swapped
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
    /// # use term_rewriting::{Context, RuleContext, Signature, parse_rule};
    /// # fn main() {
    /// let mut sig = Signature::default();
    ///
    /// let rule = parse_rule(&mut sig, "A(x_) = B(x_)").expect("parse of A(x_) = B(x_)");
    ///
    /// let new_rule = TRS::swap_lhs_and_one_rhs_helper(&rule);
    ///
    /// if new_rule == None {
    ///     assert!(false);
    /// } else {
    ///     assert_eq!(new_rule.unwrap().display(&sig), "B(x_) = A(x_)");
    /// }
    /// # }
    /// ```
    pub fn swap_lhs_and_one_rhs_helper(rule: &Rule) -> Option<Rule> {
        let r = rule.clone();
        let rhs = match r.rhs() {
            Some(rh) => rh,
            None => {
                return None;
            }
        };
        if rhs.variables().len() == r.lhs.variables().len() {
            let new_rhs = vec![r.lhs];
            return Rule::new(rhs, new_rhs);
        }
        return None;
    }
    /// swap lhs and rhs all
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate rand;
    /// # extern crate term_rewriting;
    /// # extern crate itertools;
    /// # use programinduction::trs::{TRS, Lexicon};
    /// # use rand::{thread_rng};
    /// # use term_rewriting::{Context, RuleContext, Signature, parse_rule};
    /// # use itertools::Itertools;
    /// # fn main() {
    /// let mut sig = Signature::default();
    ///
    /// let rule = parse_rule(&mut sig, "A(x_) = B(x_) | C(x_)").expect("parse of A(x_) = B(x_) | C(x_)");
    ///
    /// let new_rules = TRS::swap_lhs_and_all_rhs_helper(&rule);
    /// if new_rules == None {
    ///     assert!(false);
    /// } else {
    ///     let rules = new_rules.unwrap().iter().map(|r| format!("{};", r.display(&sig))).join("\n");
    ///     assert_eq!(rules, "B(x_) = A(x_);\nC(x_) = A(x_);");
    /// }
    /// # }
    /// ```
    pub fn swap_lhs_and_all_rhs_helper(rule: &Rule) -> Option<Vec<Rule>> {
        let mut rules: Vec<Rule> = vec![];
        let num_vars = rule.variables().len();
        for idx in 0..rule.len() {
            if rule.rhs[idx].variables().len() == num_vars {
                let lhs = rule.rhs[idx].clone();
                let rhs = vec![rule.lhs.clone()];
                let temp_rule = Rule::new(lhs, rhs);
                if temp_rule != None {
                    rules.push(temp_rule.unwrap());
                }
            }
        }
        if rules.len() == 0 {
            return None;
        }
        return Some(rules);
    }
    pub fn swap_lhs_and_rhs<R: Rng>(&self, rng: &mut R) -> Result<TRS, SampleError> {
        let mut trs = self.clone();
        let num_rules = self.len();
        let num_background = self.lex.0.read().expect("poisoned lexicon").background.len();
        if num_background >= num_rules - 1 {
            return Ok(trs);
        }
        let idx: usize = rng.gen_range(num_background, num_rules);
        let rule = trs.utrs.remove_idx(idx).expect("removing original rule");
        let rules = TRS::swap_lhs_and_all_rhs_helper(&rule);
        if rules == None {
            return Ok(trs);
        }
        trs.utrs.inserts_idx(idx, rules.unwrap()).expect("inserting rules back into trs");
        Ok(trs)
    }
    /// local difference, remove all the same
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
    /// # use term_rewriting::{Context, RuleContext, Signature, parse_rule};
    /// # fn main() {
    /// let mut sig = Signature::default();
    ///
    /// let r = parse_rule(&mut sig, "A(B C(x_)) = A(D E(x_))").expect("parse of A(B C(x_)) = A(D C(x_))");
    ///
    /// let result = TRS::local_difference_helper(&r);
    ///
    /// if result == None {
    ///     assert!(false);
    /// } else {
    ///     let rules = result.unwrap();
    ///     assert_eq!(rules[0].display(&sig), "B = D");
    ///     assert_eq!(rules[1].display(&sig), "C(x_) = E(x_)");
    /// }
    /// # }
    /// ```
    pub fn local_difference_helper(rule: &Rule) -> Option<Vec<Rule>> {
        let r = rule.clone();
        let rhs = r.rhs();
        if rhs == None {
            return None;
        }
        let temp_differences = TRS::find_differences(r.lhs, rhs.unwrap());
        if temp_differences == None {
            return None;
        }
        let differences = temp_differences.unwrap();
        let mut rules: Vec<Rule> = vec![];
        for idx in 0..differences.len() {
            let temp_rule = Rule::new(differences[idx].0.clone(), vec![differences[idx].1.clone()]);
            if temp_rule != None {
                rules.push(temp_rule.unwrap());
            }
        }
        if rules == vec![] {
            return None;
        }
        Some(rules)
    }
    pub fn find_differences(lhs: Term, rhs: Term) -> Option<Vec<(Term, Term)>> {
        if lhs == rhs {
            return None;
        }
        match lhs.clone() {
            Term::Variable(_x) => {
                return None;
            }
            Term::Application {
                op: lop,
                args: largs,
            } => {
                if largs.len() == 0 {
                    return Some(vec![(lhs, rhs)]);
                }
                match rhs.clone() {
                    Term::Variable(_x) => {
                        return Some(vec![(lhs, rhs)]);
                    }
                    Term::Application {
                        op: rop,
                        args: rargs,
                    } => {
                        if lop != rop {
                            return Some(vec![(lhs, rhs)]);
                        }
                        let mut differences: Vec<(Term, Term)> = vec![];
                        for idx in 0..largs.len() {
                            let diff =
                                TRS::find_differences(largs[idx].clone(), rargs[idx].clone());
                            if diff != None {
                                let new_diffs = diff.unwrap();
                                for ids in 0..new_diffs.len() {
                                    differences.push(new_diffs[ids].clone());
                                }
                            }
                        }
                        if differences == vec![] {
                            return None;
                        }
                        return Some(differences);
                    }
                }
            }
        }
    }
    // helper for routinization
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
    ///
    /// let result = TRS::inverse_evaluation_helper(&r, &t);
    ///
    /// assert_eq!(result.display(&sig), "A(C(y_))");
    /// # }
    /// ```
    pub fn inverse_evaluation_helper(rule: &Rule, t: &Term) -> Term {
        TRS::replace_term_helper(t, &rule.rhs[0], rule.lhs.clone())
    }
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
    ///
    /// let result = TRS::inverse_evaluate_rule_helper(&rule, &r);
    ///
    /// assert_eq!(result.unwrap().pretty(&sig), "A(C(y_)) = D C(y_)");
    /// # }
    /// ```
    pub fn inverse_evaluate_rule_helper(rule: &Rule, r: &Rule) -> Option<Rule> {
        let lhs = TRS::inverse_evaluation_helper(rule, &r.lhs);
        let mut rhs: Vec<Term> = vec![];
        for idx in 0..r.rhs.len() {
            rhs.push(TRS::inverse_evaluation_helper(rule, &r.rhs[idx]));
        }
        Rule::new(lhs, rhs)
    }
    pub fn inverse_evaluate<R: Rng>(&self, rng: &mut R) -> Result<TRS, SampleError> {
        let mut trs = self.clone();
        let num_rules = self.len();
        let num_background = self.lex.0.read().expect("poisoned lexicon").background.len();
        if num_background >= num_rules - 1 {
            return Ok(trs);
        }
        let ref_idx: usize = rng.gen_range(0, num_rules);
        let mut target_idx: usize = rng.gen_range(num_background, num_rules);
        while ref_idx == target_idx {
            target_idx = rng.gen_range(num_background, num_rules);
        }
        let new_rule = TRS::inverse_evaluate_rule_helper(&trs.utrs.rules[ref_idx], &trs.utrs.rules[target_idx]);
        if new_rule == None {
            return Ok(trs);
        }
        trs.utrs.remove_idx(target_idx).expect("removing old rule");
        trs.utrs.insert_idx(target_idx, new_rule.unwrap()).expect("inserting new rule");
        Ok(trs)
    }
    // generalizes a rule by one step, converts one constant that exists in both sides of a rule into a variable
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
    /// # use term_rewriting::{Context, RuleContext, Signature, parse_rule};
    /// # fn main() {
    /// let mut sig = Signature::default();
    ///
    /// let r = parse_rule(&mut sig, "A(B C) = A(B D)").expect("parse of A(B C) = A(B D)");
    ///
    /// let result = TRS::generalize_rule_helper(&r, &mut sig);
    ///
    /// if result == None {
    ///     assert!(false);
    /// } else {
    ///     let new_rule = result.unwrap();
    ///     assert_eq!(new_rule.display(&sig), "A(var0_ C) = A(var0_ D)");
    /// }
    /// # }
    /// ```
    pub fn generalize_rule_helper(rule: &Rule, sig: &mut Signature) -> Option<Rule> {
        let r = rule.clone();
        let lops = r.lhs.operators();
        let mut possible_consts: Vec<Operator> = vec![];
        for idx in 0..lops.len() {
            if lops[idx].arity(sig) == 0 {
                possible_consts.push(lops[idx]);
            }
        }
        if possible_consts == vec![] {
            return Some(r);
        }
        for idx in 0..possible_consts.len() {
            for ridx in 0..r.rhs.len() {
                if r.rhs[ridx].operators().contains(&possible_consts[idx]) {
                    let t = Term::Application {
                        op: possible_consts[idx].clone(),
                        args: vec![],
                    };
                    let v = Term::Variable(sig.new_var(None));
                    return TRS::replace_term_in_rule_helper(&r, &t, v);
                }
            }
        }
        Some(r)
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
