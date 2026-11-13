"""Symbolic check of the Kaluza-Klein coefficient in Proposition A.2 (Appendix A.9).

Total metric: e^{2f}(dx^2 + dy^2) + phi^2 (dz + A1 dx + A2 dy)^2, i.e. G = U(1) with q = phi^2.
Checks, on horizontal lifts, Ric_base = Ric_hat^hor + (1/2) q F_{mu a} F_nu^a
(Ricci positive on round spheres). Prints a zero matrix when the identity holds.
"""
import sympy as sp
x,y,z,phi=sp.symbols('x y z phi',real=True)
f=sp.Function('f')(x,y); A1=sp.Function('A1')(x,y); A2=sp.Function('A2')(x,y)
X=[x,y,z]
# total metric: e^{2f}(dx^2+dy^2) + phi^2 (dz + A1 dx + A2 dy)^2 ; q = phi^2
th=[A1,A2,1]
G=sp.Matrix(3,3,lambda i,j:(sp.exp(2*f) if (i==j and i<2) else 0)+phi**2*th[i]*th[j])
Gi=sp.simplify(G.inv())
def Gam(a,b,c): return sum(Gi[a,d]*(sp.diff(G[d,b],X[c])+sp.diff(G[d,c],X[b])-sp.diff(G[b,c],X[d])) for d in range(3))/2
Ga=[[[sp.simplify(Gam(a,b,c)) for c in range(3)] for b in range(3)] for a in range(3)]
def Ric(b,c):
    return sp.simplify(sum(sp.diff(Ga[a][b][c],X[a])-sp.diff(Ga[a][b][a],X[c])+sum(Ga[a][a][d]*Ga[d][b][c]-Ga[a][c][d]*Ga[d][b][a] for d in range(3)) for a in range(3)))
R=sp.Matrix(3,3,lambda i,j:Ric(i,j))
# horizontal lift of d/dx : h = d/dx - A1 d/dz
h=[sp.Matrix([1,0,-A1]),sp.Matrix([0,1,-A2])]
Rh=sp.Matrix(2,2,lambda i,j:sp.simplify((h[i].T*R*h[j])[0]))
# base Ricci for 2D conformal metric: Ric = -(f_xx+f_yy) g
gb=sp.exp(2*f)*sp.eye(2); Rb=-(sp.diff(f,x,2)+sp.diff(f,y,2))*sp.eye(2)
F=sp.diff(A2,x)-sp.diff(A1,y)
Fm=sp.Matrix([[0,F],[-F,0]]); gbi=gb.inv()
FF=Fm*gbi*Fm.T   # F_{mu a} F_{nu}^{a}
print(sp.simplify(Rb-Rh-sp.Rational(1,2)*phi**2*FF))
