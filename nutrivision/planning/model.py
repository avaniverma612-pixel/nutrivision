"""A tiny solver-agnostic integer linear model.

The meal-plan constraints are written once against this class and can then be
solved with OR-Tools CP-SAT (preferred) or SciPy's HiGHS MILP solver (fallback,
since SciPy is already a dependency of ultralytics). All coefficients are integers
so the same model is valid for CP-SAT.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Solution:
    status: str               # "optimal", "feasible", "infeasible", "error"
    values: dict[str, int]
    objective: float | None
    solver: str
    seconds: float

    @property
    def ok(self) -> bool:
        return self.status in ("optimal", "feasible")


@dataclass
class LinearModel:
    names: list[str] = field(default_factory=list)
    lbs: list[int] = field(default_factory=list)
    ubs: list[int] = field(default_factory=list)
    constraints: list[tuple[dict[int, int], float, float]] = field(default_factory=list)
    objective: dict[int, int] = field(default_factory=dict)       # maximised
    _index: dict[str, int] = field(default_factory=dict)

    def var(self, name: str, lb: int = 0, ub: int = 1) -> int:
        if name in self._index:
            raise KeyError(f"duplicate variable {name}")
        self._index[name] = len(self.names)
        self.names.append(name)
        self.lbs.append(int(lb))
        self.ubs.append(int(ub))
        return self._index[name]

    def add(self, coeffs: dict[int, int], lo: float = -np.inf, hi: float = np.inf) -> None:
        coeffs = {k: int(v) for k, v in coeffs.items() if v}
        if coeffs:
            self.constraints.append((coeffs, lo, hi))

    def add_objective(self, var: int, coef: int) -> None:
        self.objective[var] = self.objective.get(var, 0) + int(coef)

    # ------------------------------------------------------------------
    def solve(self, time_limit: float = 15.0, prefer: str = "auto") -> Solution:
        if prefer in ("auto", "cpsat"):
            try:
                return self._solve_cpsat(time_limit)
            except ImportError:
                if prefer == "cpsat":
                    raise
        return self._solve_highs(time_limit)

    def _solve_cpsat(self, time_limit: float) -> Solution:
        from ortools.sat.python import cp_model

        start = time.perf_counter()
        m = cp_model.CpModel()
        xs = [m.new_int_var(lb, ub, n) for n, lb, ub in zip(self.names, self.lbs, self.ubs)]
        big = cp_model.INT32_MAX
        for coeffs, lo, hi in self.constraints:
            expr = sum(c * xs[i] for i, c in coeffs.items())
            lo_i = -big if lo == -np.inf else int(np.ceil(lo))
            hi_i = big if hi == np.inf else int(np.floor(hi))
            m.add_linear_constraint(expr, lo_i, hi_i)
        m.maximize(sum(c * xs[i] for i, c in self.objective.items()))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_workers = 8
        code = solver.solve(m)
        status = {cp_model.OPTIMAL: "optimal", cp_model.FEASIBLE: "feasible",
                  cp_model.INFEASIBLE: "infeasible"}.get(code, "error")
        values = {n: int(solver.value(x)) for n, x in zip(self.names, xs)} if status in ("optimal", "feasible") else {}
        obj = solver.objective_value if values else None
        return Solution(status, values, obj, "OR-Tools CP-SAT", time.perf_counter() - start)

    def _solve_highs(self, time_limit: float) -> Solution:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix

        start = time.perf_counter()
        n = len(self.names)
        c = np.zeros(n)
        for i, coef in self.objective.items():
            c[i] = -coef                                 # milp minimises
        rows, cols, data, lo, hi = [], [], [], [], []
        for r, (coeffs, l, h) in enumerate(self.constraints):
            for i, coef in coeffs.items():
                rows.append(r); cols.append(i); data.append(coef)
            lo.append(l); hi.append(h)
        A = coo_matrix((data, (rows, cols)), shape=(len(self.constraints), n)).tocsr()
        res = milp(c, constraints=LinearConstraint(A, lo, hi),
                   integrality=np.ones(n), bounds=Bounds(self.lbs, self.ubs),
                   options={"time_limit": time_limit, "mip_rel_gap": 0.005, "disp": False})
        if res.x is None:
            status = "infeasible" if res.status == 2 else "error"
            return Solution(status, {}, None, "HiGHS (SciPy)", time.perf_counter() - start)
        status = "optimal" if res.status == 0 else "feasible"
        values = {name: int(round(v)) for name, v in zip(self.names, res.x)}
        return Solution(status, values, -res.fun, "HiGHS (SciPy)", time.perf_counter() - start)
