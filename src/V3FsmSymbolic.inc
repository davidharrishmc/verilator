// -*- mode: C++; c-file-style: "cc-mode" -*-
//*************************************************************************
// PROTOTYPE: symbolic FSM extraction, included from V3FsmDetect.cpp
//
// For each variable written by a clocked always block, compute its next value as a
// decision DAG over the block's statements: an assignment sets the value, an if, case
// or ?: splits it, a path that does not write the variable holds it, and statements that
// don't write it are ignored.  Reads of a next-state variable written by a combinational
// block or continuous assignment, and of compiler temporaries, resolve through their own
// DAGs.  The register is an FSM if, for every state value s, evaluating the DAG with the
// state equal to s reaches only constants and holds, and some condition depends on the
// state.  Its arcs are s -> each constant reached.  A top-level split whose one side is a
// constant regardless of the state, on a condition that does not read the state, is the
// reset.
//*************************************************************************

class FsmSymVal;
// A condition, with the values that variables it reads had when it was evaluated
struct FsmSymCond final {
    AstNodeExpr* exprp = nullptr;  // nullptr: unknown condition
    std::vector<std::pair<AstVarScope*, const FsmSymVal*>> bindings;
};

class FsmSymVal final {
public:
    enum class Kind : uint8_t { CONST, OPAQUE, REF, MUX };
    Kind kind = Kind::OPAQUE;
    V3Number num{static_cast<AstNode*>(nullptr), 1, 0};  // CONST value, as assigned
    string label;  // CONST: parameter name the value came from, if any
    FileLine* flp = nullptr;  // CONST: assignment location
    AstVarScope* refVscp = nullptr;  // REF: read of a variable's value before the block
    const FsmSymCond* condp = nullptr;  // MUX: condition, evaluated per state
    const FsmSymVal* thenp = nullptr;  // MUX: value when the condition is true
    const FsmSymVal* elsep = nullptr;  // MUX: value when false
};

class FsmSymbolicExtractor final {
    using Env = std::map<AstVarScope*, const FsmSymVal*>;
    // Leaf of a DAG evaluated for one state: a constant, a hold, or something else
    struct Leaf final {
        enum class Kind : uint8_t { CONST, HOLD, OPAQUE };
        Kind kind;
        V3Number num;
        FileLine* flp;
        bool cond;  // Reached through a condition that doesn't depend on the state alone
    };
    enum class Tri : uint8_t { FALSE, TRUE, UNKNOWN };

    FsmState& m_state;
    std::deque<FsmSymVal> m_vals;  // Arena of DAG nodes
    std::deque<FsmSymCond> m_conds;  // Arena of conditions
    std::vector<AstNode*> m_ownedConds;  // Synthesized case-item conditions, deleted at end
    const FsmSymVal* m_opaquep = nullptr;  // Shared OPAQUE leaf
    std::map<AstVarScope*, const FsmSymVal*> m_refs;  // Shared REF leaves
    // Per scope
    std::map<AstVarScope*, AstNodeExpr*> m_aliases;  // Continuous assignments, for conditions
    std::map<AstVarScope*, const FsmSymVal*> m_combVals;  // Combinationally driven values
    std::map<AstVarScope*, int> m_clockedWriters;  // Clocked blocks writing each variable
    // Statistics for the prototype report
    size_t m_regsSeen = 0;

    // DAG construction
    const FsmSymVal* opaque() {
        if (!m_opaquep) {
            m_vals.emplace_back();
            m_opaquep = &m_vals.back();
        }
        return m_opaquep;
    }
    const FsmSymVal* ref(AstVarScope* vscp) {
        const auto it = m_refs.find(vscp);
        if (it != m_refs.end()) return it->second;
        m_vals.emplace_back();
        FsmSymVal& val = m_vals.back();
        val.kind = FsmSymVal::Kind::REF;
        val.refVscp = vscp;
        m_refs.emplace(vscp, &val);
        return &val;
    }
    const FsmSymVal* constant(const V3Number& num, FileLine* flp, const string& label) {
        m_vals.emplace_back();
        FsmSymVal& val = m_vals.back();
        val.kind = FsmSymVal::Kind::CONST;
        val.num = num;
        val.flp = flp;
        val.label = label;
        return &val;
    }
    // Variables the block being walked writes with '<=': a read in the block sees their old value
    std::set<const AstVarScope*> m_dlyVscps;
    void collectDlyWrites(AstNode* stmtsp) {
        m_dlyVscps.clear();
        for (AstNode* stmtp = stmtsp; stmtp; stmtp = stmtp->nextp()) {
            stmtp->foreach([&](const AstAssignDly* assp) {
                assp->lhsp()->foreach([&](const AstVarRef* refp) {
                    if (refp->access().isWriteOrRW()) m_dlyVscps.insert(refp->varScopep());
                });
            });
        }
    }
    const FsmSymCond* makeCond(AstNodeExpr* exprp, const Env& env) {
        m_conds.emplace_back();
        FsmSymCond& cond = m_conds.back();
        cond.exprp = exprp;
        if (exprp) {
            exprp->foreach([&](const AstVarRef* refp) {
                if (m_dlyVscps.count(refp->varScopep())) return;
                const auto it = env.find(refp->varScopep());
                if (it != env.end()) cond.bindings.emplace_back(it->first, it->second);
            });
        }
        return &cond;
    }
    const FsmSymVal* mux(const FsmSymCond* condp, const FsmSymVal* thenp, const FsmSymVal* elsep) {
        if (thenp == elsep) return thenp;
        m_vals.emplace_back();
        FsmSymVal& val = m_vals.back();
        val.kind = FsmSymVal::Kind::MUX;
        val.condp = condp;
        val.thenp = thenp;
        val.elsep = elsep;
        return &val;
    }

    // A constant value of an expression (parameters, enum items, literals), if it folds
    static bool foldConst(AstNodeExpr* exprp, V3Number& num, string& label) {
        if (const AstConst* const constp = VN_CAST(exprp, Const)) {
            if (constp->num().isAnyXZ()) return false;
            num = constp->num();
            label = constp->origParamName();
            return true;
        }
        if (exprp->exists([](const AstNodeVarRef* refp) { return !refp->varp()->isParam(); })) {
            return false;
        }
        AstNode* const foldedp = V3Const::constifyEdit(exprp->cloneTree(false));
        const AstConst* const constp = VN_CAST(foldedp, Const);
        const bool ok = constp && !constp->num().isAnyXZ();
        if (ok) {
            num = constp->num();
            label = constp->origParamName();
        }
        VL_DO_DANGLING(foldedp->deleteTree(), foldedp);
        return ok;
    }

    const FsmSymVal* valueOf(AstNodeExpr* exprp, const Env& env, FileLine* flp) {
        // Width adjustments: leaves are resized to the state's width at the end
        if (AstSel* const selp = VN_CAST(exprp, Sel)) {
            const AstConst* const lsbp = VN_CAST(selp->lsbp(), Const);
            if (lsbp && lsbp->isZero()) return valueOf(selp->fromp(), env, flp);
        }
        if (AstNodeUniop* const extp = VN_CAST(exprp, Extend))
            return valueOf(extp->lhsp(), env, flp);
        if (AstCond* const condp = VN_CAST(exprp, Cond)) {
            return mux(makeCond(condp->condp(), env), valueOf(condp->thenp(), env, flp),
                       valueOf(condp->elsep(), env, flp));
        }
        if (AstVarRef* const refp = VN_CAST(exprp, VarRef)) {
            if (!refp->varp()->isParam()) {
                const auto it = m_dlyVscps.count(refp->varScopep()) ? env.end()
                                                                    : env.find(refp->varScopep());
                return it != env.end() ? it->second : ref(refp->varScopep());
            }
        }
        V3Number num{static_cast<AstNode*>(nullptr), 1, 0};
        string label;
        if (foldConst(exprp, num, label)) return constant(num, flp, label);
        return opaque();
    }

    static Env::mapped_type lookup(const Env& env, const Env& before, AstVarScope* vscp,
                                   FsmSymbolicExtractor* selfp) {
        auto it = env.find(vscp);
        if (it != env.end()) return it->second;
        it = before.find(vscp);
        if (it != before.end()) return it->second;
        return selfp->ref(vscp);
    }
    void join(const FsmSymCond* condp, const Env& before, const Env& thenEnv, const Env& elseEnv,
              Env& out) {
        std::set<AstVarScope*> keys;
        for (const auto& it : thenEnv) keys.insert(it.first);
        for (const auto& it : elseEnv) keys.insert(it.first);
        for (AstVarScope* const vscp : keys) {
            out[vscp] = mux(condp, lookup(thenEnv, before, vscp, this),
                            lookup(elseEnv, before, vscp, this));
        }
    }

    // Statements
    void writeOpaque(AstNode* nodep, Env& env) {
        nodep->foreach([&](const AstNodeVarRef* refp) {
            if (refp->access().isWriteOrRW()) env[refp->varScopep()] = opaque();
        });
    }
    AstNodeExpr* caseItemCond(AstCase* casep, AstCaseItem* itemp) {
        if (casep->caseInside()) return nullptr;
        AstNodeExpr* resultp = nullptr;
        for (AstNodeExpr* const labelp : itemp->condspList()) {
            AstNodeExpr* lhsp = casep->exprp()->cloneTree(false);
            AstNodeExpr* rhsp = labelp->cloneTree(false);
            // Compare at the wider width, as a case does
            if (lhsp->width() < rhsp->width()) {
                lhsp = new AstExtend{lhsp->fileline(), lhsp, rhsp->width()};
            } else if (rhsp->width() < lhsp->width()) {
                rhsp = new AstExtend{rhsp->fileline(), rhsp, lhsp->width()};
            }
            const AstConst* const constp = VN_CAST(labelp, Const);
            const bool wild
                = casep->casez() || casep->casex() || (constp && constp->num().isAnyXZ());
            AstNodeExpr* const eqp
                = wild ? static_cast<AstNodeExpr*>(new AstEqWild{labelp->fileline(), lhsp, rhsp})
                       : new AstEq{labelp->fileline(), lhsp, rhsp};
            resultp = resultp ? new AstLogOr{labelp->fileline(), resultp, eqp} : eqp;
        }
        if (resultp) m_ownedConds.push_back(resultp);
        return resultp;
    }
    void walk(AstNode* stmtsp, Env& env) {
        for (AstNode* stmtp = stmtsp; stmtp; stmtp = stmtp->nextp()) {
            if (AstNodeAssign* const assp = VN_CAST(stmtp, NodeAssign)) {
                if (AstVarRef* const lhsp = VN_CAST(assp->lhsp(), VarRef)) {
                    env[lhsp->varScopep()] = valueOf(assp->rhsp(), env, assp->fileline());
                } else {
                    writeOpaque(assp->lhsp(), env);
                }
            } else if (AstIf* const ifp = VN_CAST(stmtp, If)) {
                Env thenEnv;
                Env elseEnv;
                Env inner = env;
                walk(ifp->thensp(), inner);
                for (const auto& it : inner) {
                    const auto bit = env.find(it.first);
                    if (bit == env.end() || bit->second != it.second) thenEnv.insert(it);
                }
                inner = env;
                walk(ifp->elsesp(), inner);
                for (const auto& it : inner) {
                    const auto bit = env.find(it.first);
                    if (bit == env.end() || bit->second != it.second) elseEnv.insert(it);
                }
                Env out;
                join(makeCond(ifp->condp(), env), env, thenEnv, elseEnv, out);
                for (const auto& it : out) env[it.first] = it.second;
            } else if (AstCase* const casep = VN_CAST(stmtp, Case)) {
                // Items in order: the first match wins, default (or no match) last
                std::vector<std::pair<const FsmSymCond*, Env>> arms;
                Env defaultEnv = env;
                bool unknownItem = false;
                for (AstCaseItem* const itemp : casep->itemspList()) {
                    Env inner = env;
                    walk(itemp->stmtsp(), inner);
                    if (itemp->isDefault()) {
                        defaultEnv = inner;
                        continue;
                    }
                    AstNodeExpr* const condp = caseItemCond(casep, itemp);
                    if (!condp) unknownItem = true;
                    arms.emplace_back(makeCond(condp, env), inner);
                }
                if (unknownItem) {
                    writeOpaque(casep, env);
                    continue;
                }
                Env result = defaultEnv;
                for (auto it = arms.rbegin(); it != arms.rend(); ++it) {
                    Env out;
                    join(it->first, env, it->second, result, out);
                    for (const auto& kv : out) result[kv.first] = kv.second;
                }
                env = result;
            } else if (AstBegin* const beginp = VN_CAST(stmtp, Begin)) {
                walk(beginp->stmtsp(), env);
            } else {
                // Coverage probes, displays, loops, calls: only their writes matter
                writeOpaque(stmtp, env);
            }
        }
    }

    // Resolve REF leaves: the state itself stays a REF (a hold); a combinationally driven
    // variable resolves to its own DAG; anything else is opaque
    const FsmSymVal* resolve(const FsmSymVal* valp, AstVarScope* stateVscp,
                             std::map<const FsmSymVal*, const FsmSymVal*>& memo, int depth) {
        if (valp->kind == FsmSymVal::Kind::CONST || valp->kind == FsmSymVal::Kind::OPAQUE) {
            return valp;
        }
        const auto it = memo.find(valp);
        if (it != memo.end()) return it->second;
        const FsmSymVal* resultp = opaque();
        if (valp->kind == FsmSymVal::Kind::REF) {
            if (valp->refVscp == stateVscp) {
                resultp = valp;
            } else if (depth < 8) {
                const auto cit = m_combVals.find(valp->refVscp);
                if (cit != m_combVals.end()) {
                    memo[valp] = opaque();  // Cycle guard
                    resultp = resolve(cit->second, stateVscp, memo, depth + 1);
                }
            }
        } else {
            resultp = mux(valp->condp, resolve(valp->thenp, stateVscp, memo, depth),
                          resolve(valp->elsep, stateVscp, memo, depth));
        }
        memo[valp] = resultp;
        return resultp;
    }

    // Conditions: substitute the state, bound and combinationally driven variables, and
    // continuous aliases, then fold
    // A condition, and the undecided conditions taken on the path to it
    using SeenKey = std::pair<const FsmSymCond*, std::vector<std::pair<const FsmSymCond*, bool>>>;
    struct EvalCtx final {
        AstVarScope* stateVscp;
        V3Number s;
        int width;
        std::map<const FsmSymCond*, Tri> condMemo;
        // Results per condition and path, over all states
        std::map<SeenKey, std::set<Tri>>* seenp;
        std::map<const FsmSymVal*, const FsmSymVal*> resolveMemo;
        std::map<const FsmSymCond*, bool> path;  // Undecided conditions taken on this path
        const FsmSymCond* splitp = nullptr;  // An undecided condition to split the path on
        int pathUses = 0;  // Lookups of 'path': a result that used one isn't memoized
    };
    // The single value of 'valp' in this state, if it has one
    bool singleValue(const FsmSymVal* valp, EvalCtx& ctx, V3Number& num, int depth) {
        if (depth > 6) return false;
        valp = resolve(valp, ctx.stateVscp, ctx.resolveMemo, 0);
        std::vector<Leaf> leaves;
        evaluate(valp, ctx, false, leaves, depth + 1);
        bool found = false;
        for (const Leaf& leaf : leaves) {
            if (leaf.kind == Leaf::Kind::OPAQUE) return false;
            const V3Number& lnum = leaf.kind == Leaf::Kind::HOLD ? ctx.s : leaf.num;
            if (found && !(lnum.width() == num.width() && lnum.isCaseEq(num))) return false;
            num = lnum;
            found = true;
        }
        return found;
    }
    AstNodeExpr* substitute(AstNodeExpr* exprp, const FsmSymCond& cond, EvalCtx& ctx, int depth) {
        if (AstVarRef* const refp = VN_CAST(exprp, VarRef)) {
            AstVarScope* const vscp = refp->varScopep();
            V3Number num{refp, refp->width()};
            bool known = false;
            const FsmSymVal* valp = nullptr;
            for (const auto& binding : cond.bindings) {
                if (binding.first == vscp) valp = binding.second;
            }
            if (!valp && vscp == ctx.stateVscp) {
                num.opAssign(ctx.s);
                known = true;
            } else {
                if (!valp) {
                    const auto it = m_combVals.find(vscp);
                    if (it != m_combVals.end()) valp = it->second;
                }
                V3Number value{refp, refp->width()};
                if (valp && singleValue(valp, ctx, value, depth)) {
                    num.opAssign(toWidth(value, refp->width()));
                    known = true;
                }
            }
            if (known) {
                AstConst* const constp = new AstConst{refp->fileline(), num};
                VL_DO_DANGLING(refp->deleteTree(), refp);
                return constp;
            }
            const auto it = m_aliases.find(vscp);
            if (it != m_aliases.end() && depth < 4 && it->second->width() == refp->width()) {
                AstNodeExpr* const clonep = it->second->cloneTree(false);
                VL_DO_DANGLING(refp->deleteTree(), refp);
                return substitute(clonep, FsmSymCond{}, ctx, depth + 1);
            }
            return refp;
        }
        std::vector<AstVarRef*> refps;
        exprp->foreach([&](AstVarRef* refp) { refps.push_back(refp); });
        for (AstVarRef* const refp : refps) {
            AstNodeExpr* const newp = substitute(refp->cloneTree(false), cond, ctx, depth);
            refp->replaceWith(newp);
            VL_DO_DANGLING(refp->deleteTree(), refp);
        }
        return exprp;
    }
    // Bits known after substitution: constants, including inside concatenations and zero
    // extensions (-1 unknown, 0, 1, 2 for X/Z)
    static void knownBits(const AstNodeExpr* exprp, std::vector<int>& bits, int lsb) {
        if (const AstConst* const constp = VN_CAST(exprp, Const)) {
            for (int i = 0; i < constp->width(); ++i) {
                bits[lsb + i] = constp->num().bitIs1(i) ? 1 : constp->num().bitIs0(i) ? 0 : 2;
            }
        } else if (const AstConcat* const catp = VN_CAST(exprp, Concat)) {
            knownBits(catp->rhsp(), bits, lsb);
            knownBits(catp->lhsp(), bits, lsb + catp->rhsp()->width());
        } else if (const AstExtend* const extp = VN_CAST(exprp, Extend)) {
            for (int i = extp->lhsp()->width(); i < extp->width(); ++i) bits[lsb + i] = 0;
            knownBits(extp->lhsp(), bits, lsb);
        }
    }
    // An equality of partly known values, e.g. {state, in} once the state is substituted, that
    // V3Const leaves alone: a known bit that differs decides it
    static AstNodeExpr* foldKnownBits(AstNodeExpr* condp) {
        std::vector<AstNodeBiop*> cmps;
        condp->foreach([&](AstNodeBiop* biopp) {
            if (VN_IS(biopp, Eq) || VN_IS(biopp, EqWild) || VN_IS(biopp, EqCase)
                || VN_IS(biopp, Neq) || VN_IS(biopp, NeqWild) || VN_IS(biopp, NeqCase)) {
                cmps.push_back(biopp);
            }
        });
        // Innermost first, so replacing one never frees another still to visit
        for (auto it = cmps.rbegin(); it != cmps.rend(); ++it) {
            AstNodeBiop* const cmpp = *it;
            const int width = cmpp->lhsp()->width();
            if (cmpp->rhsp()->width() != width) continue;
            std::vector<int> lbits(width, -1);
            std::vector<int> rbits(width, -1);
            knownBits(cmpp->lhsp(), lbits, 0);
            knownBits(cmpp->rhsp(), rbits, 0);
            const bool wild = VN_IS(cmpp, EqWild) || VN_IS(cmpp, NeqWild);
            bool differ = false;
            for (int i = 0; i < width; ++i) {
                if (wild && rbits[i] == 2) continue;
                if (lbits[i] >= 0 && lbits[i] < 2 && rbits[i] >= 0 && rbits[i] < 2
                    && lbits[i] != rbits[i]) {
                    differ = true;
                }
            }
            if (!differ) continue;
            const bool neq = VN_IS(cmpp, Neq) || VN_IS(cmpp, NeqWild) || VN_IS(cmpp, NeqCase);
            AstConst* const newp = neq ? new AstConst{cmpp->fileline(), AstConst::BitTrue{}}
                                       : new AstConst{cmpp->fileline(), AstConst::BitFalse{}};
            if (cmpp == condp) {
                VL_DO_DANGLING(cmpp->deleteTree(), cmpp);
                return newp;
            }
            cmpp->replaceWith(newp);
            VL_DO_DANGLING(cmpp->deleteTree(), cmpp);
        }
        return condp;
    }
    // The first undecided condition that chooses among several values of 'valp' on this path
    const FsmSymCond* undecidedCond(const FsmSymVal* valp, EvalCtx& ctx, int depth) {
        V3Number one{static_cast<AstNode*>(nullptr), 1, 0};
        if (depth > 6 || singleValue(valp, ctx, one, depth)) return nullptr;
        valp = resolve(valp, ctx.stateVscp, ctx.resolveMemo, 0);
        while (valp->kind == FsmSymVal::Kind::MUX) {
            const FsmSymCond* const savedp = ctx.splitp;
            ctx.splitp = nullptr;
            const Tri result = evalCond(valp->condp, ctx, depth + 1);
            const FsmSymCond* const innerp = ctx.splitp;
            ctx.splitp = savedp;
            if (result == Tri::TRUE) {
                valp = valp->thenp;
            } else if (result == Tri::FALSE) {
                valp = valp->elsep;
            } else {
                return innerp ? innerp : valp->condp;
            }
        }
        return nullptr;
    }
    Tri evalOnce(const FsmSymCond* condp, EvalCtx& ctx, int depth) {
        AstNodeExpr* const clonep
            = foldKnownBits(substitute(condp->exprp->cloneTree(false), *condp, ctx, depth));
        AstNode* const foldedp = V3Const::constifyEdit(clonep);
        Tri result = Tri::UNKNOWN;
        if (const AstConst* const constp = VN_CAST(foldedp, Const)) {
            if (!constp->num().isAnyXZ()) {
                result = constp->num().isNeqZero() ? Tri::TRUE : Tri::FALSE;
            }
        }
        VL_DO_DANGLING(foldedp->deleteTree(), foldedp);
        return result;
    }
    Tri evalCond(const FsmSymCond* condp, EvalCtx& ctx, int depth) {
        if (!condp->exprp) return Tri::UNKNOWN;
        const auto pit = ctx.path.find(condp);
        if (pit != ctx.path.end()) {
            ++ctx.pathUses;
            return pit->second ? Tri::TRUE : Tri::FALSE;
        }
        const auto it = ctx.condMemo.find(condp);
        if (it != ctx.condMemo.end()) return it->second;
        // A variable the block gave one of several values earlier in this cycle, e.g. the
        // state after 'if (start) state = S1;': split the path on what chooses the value
        for (const auto& binding : condp->bindings) {
            if (const FsmSymCond* const undecidedp = undecidedCond(binding.second, ctx, depth)) {
                ctx.splitp = undecidedp;
                return Tri::UNKNOWN;
            }
        }
        const int uses = ctx.pathUses;
        const Tri result = evalOnce(condp, ctx, depth);
        if (ctx.pathUses == uses) ctx.condMemo[condp] = result;
        if (ctx.seenp) {
            (*ctx.seenp)[SeenKey{condp, {ctx.path.begin(), ctx.path.end()}}].insert(result);
        }
        return result;
    }
    void evaluate(const FsmSymVal* valp, EvalCtx& ctx, bool cond, std::vector<Leaf>& leaves,
                  int depth) {
        switch (valp->kind) {
        case FsmSymVal::Kind::CONST:
            leaves.push_back(Leaf{Leaf::Kind::CONST, valp->num, valp->flp, cond});
            return;
        case FsmSymVal::Kind::REF:
            // Another register's old value is unknown here
            leaves.push_back(
                Leaf{valp->refVscp == ctx.stateVscp ? Leaf::Kind::HOLD : Leaf::Kind::OPAQUE, ctx.s,
                     nullptr, cond});
            return;
        case FsmSymVal::Kind::OPAQUE:
            leaves.push_back(Leaf{Leaf::Kind::OPAQUE, ctx.s, nullptr, cond});
            return;
        case FsmSymVal::Kind::MUX: {
            const Tri result = evalCond(valp->condp, ctx, depth);
            if (result == Tri::UNKNOWN && ctx.splitp) {
                const FsmSymCond* const splitp = ctx.splitp;
                ctx.splitp = nullptr;
                for (const bool value : {true, false}) {
                    ctx.path[splitp] = value;
                    evaluate(valp, ctx, true, leaves, depth);
                    ctx.path.erase(splitp);
                }
                return;
            }
            const bool unknown = result == Tri::UNKNOWN;
            for (const bool value : {true, false}) {
                if (result == (value ? Tri::FALSE : Tri::TRUE)) continue;
                // An undecided condition keeps its value for the rest of the path
                if (unknown) ctx.path[valp->condp] = value;
                evaluate(value ? valp->thenp : valp->elsep, ctx, cond || unknown, leaves, depth);
                if (unknown) ctx.path.erase(valp->condp);
            }
            return;
        }
        }
    }

    static V3Number toWidth(const V3Number& num, int width) {
        V3Number result{static_cast<AstNode*>(nullptr), width, 0};
        if (num.width() > width) {
            result.opSel(num, width - 1, 0);
        } else {
            result.opAssign(num);  // Zero extends
        }
        return result;
    }

    // Try one register written by a clocked block
    void tryRegister(AstScope* scopep, AstAlways* alwaysp, AstVarScope* stateVscp,
                     const FsmSymVal* nextp) {
        ++m_regsSeen;
        const int width = stateVscp->width();
        if (width >= 64) return;
        std::map<const FsmSymVal*, const FsmSymVal*> memo;
        const FsmSymVal* treep = resolve(nextp, stateVscp, memo, 0);
        if (treep->kind != FsmSymVal::Kind::MUX) return;
        AstEnumDType* const enump
            = VN_CAST(stateVscp->varp()->dtypep()->skipRefToEnump(), EnumDType);
        const bool declared = enump || stateVscp->varp()->attrFsmState();
        // A one-bit register is a flag unless declared as a state (as in Yosys fsm_detect)
        if (!declared && width < 2) return;
        // State space: enum items, else the constants the logic assigns or compares
        std::vector<std::pair<string, V3Number>> states;
        std::set<string> keys;
        const auto addState = [&](const string& label, const V3Number& num) {
            const V3Number wnum = toWidth(num, width);
            if (keys.insert(wnum.ascii(false, true)).second) {
                states.emplace_back(label.empty() ? wnum.ascii() : label, wnum);
                return;
            }
            // A parameter name is a better label than the value
            if (label.empty()) return;
            for (auto& state : states) {
                if (state.second.isCaseEq(wnum) && state.first == wnum.ascii())
                    state.first = label;
            }
        };
        const auto stateIndex = [&](const V3Number& num) {
            const V3Number wnum = toWidth(num, width);
            for (size_t i = 0; i < states.size(); ++i) {
                if (states[i].second.isCaseEq(wnum)) return i;
            }
            return states.size();
        };
        // Values the logic gives the register or compares it with (as in the other tools,
        // an enum value the logic never names isn't a state)
        std::vector<std::pair<string, V3Number>> named;
        {
            // A variable that holds the state: the state itself, or a copy of it
            const auto isStateRef = [&](const AstVarRef* refp, const FsmSymCond* condp) {
                AstVarScope* const vscp = refp->varScopep();
                if (vscp == stateVscp) return true;
                const FsmSymVal* valp = nullptr;
                for (const auto& binding : condp->bindings) {
                    if (binding.first == vscp) valp = binding.second;
                }
                if (!valp) {
                    const auto it = m_combVals.find(vscp);
                    if (it != m_combVals.end()) valp = it->second;
                }
                if (valp && valp->kind == FsmSymVal::Kind::REF && valp->refVscp == stateVscp) {
                    return true;
                }
                const auto ait = m_aliases.find(vscp);
                const AstVarRef* const aliasp
                    = ait == m_aliases.end() ? nullptr : VN_CAST(ait->second, VarRef);
                return aliasp && aliasp->varScopep() == stateVscp;
            };
            // Bit offset of the state in 'nodep': the state itself, zero-extended, or a field of
            // a concatenation; -1 if it is none of these
            std::function<int(const AstNode*, const FsmSymCond*)> stateLsb;
            stateLsb = [&](const AstNode* nodep, const FsmSymCond* condp) -> int {
                if (const AstVarRef* const refp = VN_CAST(nodep, VarRef)) {
                    return isStateRef(refp, condp) ? 0 : -1;
                }
                if (const AstExtend* const extp = VN_CAST(nodep, Extend)) {
                    return stateLsb(extp->lhsp(), condp);
                }
                if (const AstConcat* const catp = VN_CAST(nodep, Concat)) {
                    const int low = stateLsb(catp->rhsp(), condp);
                    if (low >= 0) return low;
                    const int high = stateLsb(catp->lhsp(), condp);
                    return high >= 0 ? high + catp->rhsp()->width() : -1;
                }
                return -1;
            };
            std::function<void(AstNodeExpr*, const FsmSymCond*, int)> compares;
            compares = [&](AstNodeExpr* exprp, const FsmSymCond* condp, int depth) {
                exprp->foreach([&](const AstNodeBiop* biopp) {
                    if (!VN_IS(biopp, Eq) && !VN_IS(biopp, EqCase) && !VN_IS(biopp, EqWild)
                        && !VN_IS(biopp, Neq) && !VN_IS(biopp, NeqCase)
                        && !VN_IS(biopp, NeqWild)) {
                        return;
                    }
                    for (const AstNode* const sidep : {biopp->lhsp(), biopp->rhsp()}) {
                        const AstConst* const constp = VN_CAST(sidep, Const);
                        const AstNode* const otherp
                            = sidep == biopp->lhsp() ? biopp->rhsp() : biopp->lhsp();
                        const int lsb = constp ? stateLsb(otherp, condp) : -1;
                        if (lsb < 0 || constp->width() < lsb + width) continue;
                        // The state's field, e.g. S of {state, in} ==? {S, 2'b?1}
                        V3Number field{static_cast<AstNode*>(nullptr), width, 0};
                        field.opSel(constp->num(), lsb + width - 1, lsb);
                        if (field.isAnyXZ()) continue;
                        named.emplace_back(
                            VN_IS(otherp, Concat) ? string{} : constp->origParamName(), field);
                    }
                });
                // Compares inside continuously assigned wires the condition reads
                if (depth >= 4) return;
                exprp->foreach([&](const AstVarRef* refp) {
                    const auto it = m_aliases.find(refp->varScopep());
                    if (it != m_aliases.end()) compares(it->second, condp, depth + 1);
                });
            };
            std::vector<const FsmSymVal*> stack{treep};
            std::set<const FsmSymVal*> seen;
            while (!stack.empty()) {
                const FsmSymVal* const valp = stack.back();
                stack.pop_back();
                if (!seen.insert(valp).second) continue;
                if (valp->kind == FsmSymVal::Kind::CONST) {
                    named.emplace_back(valp->label, valp->num);
                } else if (valp->kind == FsmSymVal::Kind::MUX) {
                    if (valp->condp->exprp) compares(valp->condp->exprp, valp->condp, 0);
                    stack.push_back(valp->thenp);
                    stack.push_back(valp->elsep);
                }
            }
        }
        if (enump) {
            std::set<string> namedKeys;
            for (const auto& it : named) {
                namedKeys.insert(toWidth(it.second, width).ascii(false, true));
            }
            for (AstEnumItem* const itemp : enump->itemspList()) {
                const AstConst* const constp = VN_CAST(itemp->valuep(), Const);
                if (!constp) return;
                if (namedKeys.count(toWidth(constp->num(), width).ascii(false, true))) {
                    addState(itemp->name(), constp->num());
                }
            }
        } else {
            for (const auto& it : named) addState(it.first, it.second);
        }
        if (states.size() < 2) return;
        // Evaluate the DAG with the state equal to each state value
        std::map<SeenKey, std::set<Tri>> seenResults;
        std::vector<std::vector<Leaf>> leavesByState(states.size());
        for (size_t from = 0; from < states.size(); ++from) {
            EvalCtx ctx{stateVscp, states[from].second, width, {}, &seenResults, {}};
            evaluate(treep, ctx, false, leavesByState[from], 0);
        }
        // The next state depends on the state through some condition
        bool dispatch = false;
        for (const auto& it : seenResults) dispatch |= it.second.size() > 1;
        if (!dispatch && !stateVscp->varp()->attrFsmState()) return;
        // Reset: a top-level split on a condition that is the same in every state, with a
        // value on one side that doesn't depend on the state
        const FsmSymVal* resetp = nullptr;
        const FsmSymVal* normalp = treep;
        {
            // Never split on a path, and the same result in every state
            bool stateFree = false;
            for (const auto& it : seenResults) {
                if (it.first.first != treep->condp) continue;
                stateFree = it.first.second.empty() && it.second.size() == 1;
                if (!stateFree) break;
            }
            const auto stateFreeVal = [](const FsmSymVal* valp) {
                return valp->kind == FsmSymVal::Kind::CONST
                       || valp->kind == FsmSymVal::Kind::OPAQUE;
            };
            if (stateFree && stateFreeVal(treep->thenp)) {
                resetp = treep->thenp;
                normalp = treep->elsep;
            } else if (stateFree && stateFreeVal(treep->elsep)) {
                resetp = treep->elsep;
                normalp = treep->thenp;
            }
        }
        if (resetp && resetp->kind == FsmSymVal::Kind::CONST
            && stateIndex(resetp->num) == states.size()) {
            if (enump) {
                resetp = nullptr;  // A reset value outside the enum: no reset arcs
            } else {
                addState(resetp->label, resetp->num);
            }
        }
        // Arcs, from the evaluation without the reset split
        std::vector<std::tuple<size_t, size_t, bool, FileLine*>> arcs;
        for (size_t from = 0; from < states.size(); ++from) {
            std::vector<Leaf> leaves;
            EvalCtx ctx{stateVscp, states[from].second, width, {}, nullptr, {}};
            evaluate(normalp, ctx, false, leaves, 0);
            for (const Leaf& leaf : leaves) {
                if (leaf.kind == Leaf::Kind::OPAQUE) {
                    if (declared) continue;  // Not modeled, as before
                    return;  // A register whose value isn't one of its constants
                }
                if (leaf.kind == Leaf::Kind::HOLD) continue;
                size_t to = stateIndex(leaf.num);
                if (to == states.size()) {
                    if (enump) continue;  // A value outside the enum
                    addState("", leaf.num);
                }
                if (to != from) arcs.emplace_back(from, to, leaf.cond, leaf.flp);
            }
        }
        // PROTOTYPE option: keep only arcs from states reachable from the reset state
        const char* const reachp = std::getenv("VERILATOR_FSM_SYMBOLIC_REACH");
        if (reachp && string{reachp} == "1" && resetp && resetp->kind == FsmSymVal::Kind::CONST) {
            std::set<size_t> reached{stateIndex(resetp->num)};
            for (bool grew = true; grew;) {
                grew = false;
                for (const auto& arc : arcs) {
                    if (reached.count(std::get<0>(arc))
                        && reached.insert(std::get<1>(arc)).second) {
                        grew = true;
                    }
                }
            }
            arcs.erase(std::remove_if(arcs.begin(), arcs.end(),
                                      [&](const std::tuple<size_t, size_t, bool, FileLine*>& arc) {
                                          return !reached.count(std::get<0>(arc));
                                      }),
                       arcs.end());
        }
        // An FSM whose transitions all hold still has state coverage
        // Graph, in the form the lowering expects
        DetectedFsm& entry = m_state.fsmFor(stateVscp);
        if (entry.graphp) return;
        entry.graphp.reset(new FsmGraph{});
        FsmGraph& graph = *entry.graphp;
        graph.scopep(scopep);
        graph.stateAlwaysp(alwaysp);
        graph.stateVarName(stateVscp->prettyName());
        graph.stateVarInternalName(stateVscp->varp()->name());
        graph.stateVarScopep(stateVscp);
        graph.sampleVarScopep(stateVscp);
        for (AstSenItem* const itemp : alwaysp->sentreep()->sensespList()) {
            AstNodeVarRef* const vrefp = itemp->varrefp();
            if (!vrefp) continue;
            FsmSenDesc desc;
            desc.edgeType = itemp->edgeType().m_e;
            desc.varScopep = vrefp->varScopep();
            graph.senses().push_back(desc);
        }
        graph.hasResetCond(resetp != nullptr);
        graph.inclCond(stateVscp->varp()->attrFsmArcInclCond());
        graph.fileline(alwaysp->fileline());
        for (const auto& state : states) {
            graph.addStateVertex(state.first, FsmStateValue{state.second});
        }
        if (resetp && resetp->kind == FsmSymVal::Kind::CONST) {
            graph.addArc(FsmStateValue{}, FsmStateValue{states[stateIndex(resetp->num)].second},
                         true, false, false, resetp->flp);
        }
        std::set<std::pair<size_t, size_t>> added;
        for (const auto& arc : arcs) {
            if (!added.emplace(std::get<0>(arc), std::get<1>(arc)).second) continue;
            graph.addArc(FsmStateValue{states[std::get<0>(arc)].second},
                         FsmStateValue{states[std::get<1>(arc)].second}, false, std::get<2>(arc),
                         false, std::get<3>(arc));
        }
    }

    static bool isClocked(AstAlways* alwaysp) {
        return alwaysp->sentreep() && alwaysp->sentreep()->hasEdge();
    }

    void processScope(AstScope* scopep) {
        m_aliases.clear();
        m_combVals.clear();
        m_clockedWriters.clear();
        std::vector<AstAlways*> clocked;
        for (AstNode* nodep = scopep->blocksp(); nodep; nodep = nodep->nextp()) {
            if (AstAssignW* const assp = VN_CAST(nodep, AssignW)) {
                if (AstVarRef* const lhsp = VN_CAST(assp->lhsp(), VarRef)) {
                    m_aliases[lhsp->varScopep()] = assp->rhsp();
                    m_combVals[lhsp->varScopep()] = valueOf(assp->rhsp(), Env{}, assp->fileline());
                }
            } else if (AstAlways* const alwaysp = VN_CAST(nodep, Always)) {
                if (alwaysp->keyword() == VAlwaysKwd::CONT_ASSIGN) {
                    AstNodeAssign* const assp = VN_CAST(alwaysp->stmtsp(), NodeAssign);
                    AstVarRef* const lhsp = assp ? VN_CAST(assp->lhsp(), VarRef) : nullptr;
                    if (lhsp) {
                        m_aliases[lhsp->varScopep()] = assp->rhsp();
                        m_combVals[lhsp->varScopep()]
                            = valueOf(assp->rhsp(), Env{}, assp->fileline());
                    }
                } else if (isClocked(alwaysp)) {
                    clocked.push_back(alwaysp);
                } else {
                    Env env;
                    collectDlyWrites(alwaysp->stmtsp());
                    walk(alwaysp->stmtsp(), env);
                    m_dlyVscps.clear();
                    for (const auto& it : env) m_combVals[it.first] = it.second;
                }
            }
        }
        std::vector<Env> envs(clocked.size());
        for (size_t i = 0; i < clocked.size(); ++i) {
            collectDlyWrites(clocked[i]->stmtsp());
            walk(clocked[i]->stmtsp(), envs[i]);
            m_dlyVscps.clear();
            for (const auto& it : envs[i]) ++m_clockedWriters[it.first];
        }
        for (size_t i = 0; i < clocked.size(); ++i) {
            for (const auto& it : envs[i]) {
                if (m_clockedWriters[it.first] != 1) continue;
                if (it.first->varp()->isParam()) continue;
                tryRegister(scopep, clocked[i], it.first, it.second);
            }
        }
    }

public:
    FsmSymbolicExtractor(FsmState& state, AstNetlist* rootp)
        : m_state{state} {
        rootp->foreach([&](AstScope* scopep) { processScope(scopep); });
        UINFO(2, "FSM symbolic: registers examined " << m_regsSeen);
    }
    ~FsmSymbolicExtractor() {
        for (AstNode* const nodep : m_ownedConds) VL_DO_DANGLING(nodep->deleteTree(), nodep);
    }
};

// One line per FSM with its states and arcs, for comparing extractors
string fsmSummary(const FsmGraph& graph) {
    std::vector<string> states;
    std::vector<string> arcs;
    for (const V3GraphVertex& vtx : graph.vertices()) {
        const FsmVertex* const vertexp = vtx.as<FsmVertex>();
        if (vertexp->isState()) states.push_back(vertexp->label());
        for (const V3GraphEdge& edge : vertexp->outEdges()) {
            const FsmArcEdge* const arcp = edge.as<FsmArcEdge>();
            const string from = arcp->isReset()     ? string{"@RESET"}
                                : arcp->isDefault() ? string{"@DEFAULT"}
                                                    : vertexp->label();
            arcs.push_back(from + "->" + edge.top()->as<FsmVertex>()->label());
        }
    }
    std::sort(states.begin(), states.end());
    std::sort(arcs.begin(), arcs.end());
    arcs.erase(std::unique(arcs.begin(), arcs.end()), arcs.end());
    const string modName = graph.scopep() ? graph.scopep()->modp()->prettyName() : "?";
    string result = graph.stateVarName() + " module=" + modName
                    + " states=" + cvtToStr(states.size()) + " {";
    for (const string& state : states) result += " " + state;
    result += " } arcs=" + cvtToStr(arcs.size()) + " {";
    for (const string& arc : arcs) result += " " + arc;
    return result + " }";
}
