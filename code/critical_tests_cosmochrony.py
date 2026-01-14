import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
from scipy.special import sph_harm
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

print("=" * 80)
print(" CRITICAL TESTS FOR COSMOCHRONY SPECTRAL PREDICTIONS ".center(80))
print("=" * 80)
print("\nTesting three hypotheses:")
print("  [1] Explicit torsion operator Ω_w required")
print("  [2] S³ geometry required (not flat Euclidean)")
print("  [3] Potential V(χ) provides fine corrections")
print("=" * 80)


# ============================================================================
# TEST #1: EXPLICIT TORSION OPERATOR Ω_w
# ============================================================================

class CosmoChronyNetworkWithTorsion:
  """
  Extension avec opérateur de torsion explicite Ω_w
  qui ne commute pas avec ∆^(0)_G pour w ≥ 2
  """

  def __init__(self, N=512, chi_c=1.0, K0=1.0):
    n = int(np.cbrt(N))
    self.nodes = n ** 3
    self.chi_c = chi_c
    self.K0 = K0

    # Réseau cubique 3D
    x, y, z = np.meshgrid(np.arange(n), np.arange(n), np.arange(n))
    self.positions = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    # Champ χ initial
    self.chi = np.random.randn(self.nodes) * 0.2

    # Centre pour les excitations
    self.center = self.positions[self.nodes // 2]

  def build_laplacian_with_torsion(self, winding_number=0, torsion_strength=0.5):
    """
    Construit ∆^(0)_G + correction de torsion Ω_w

    Args:
        winding_number: w ∈ {0, 1, 2, 3}
        torsion_strength: Intensité du couplage [∆, Ω]
    """
    # Laplacien de base
    chi_bar = self._compute_background_field()
    K_matrix = self._build_coupling_matrix(chi_bar)
    degree = np.array(K_matrix.sum(axis=1)).flatten()
    D = diags(degree, 0, format='csr')
    L_base = D - K_matrix

    if winding_number == 0:
      return L_base

    # Opérateur de torsion Ω_w
    Omega = self._build_torsion_operator(winding_number)

    # Commutateur [∆, Ω] = ∆Ω - Ω∆
    # Pour w ≥ 2, ce commutateur est non-nul → frustration spectrale
    commutator = L_base @ Omega - Omega @ L_base

    # Laplacien modifié (Eq. 37 du document)
    # L_twisted = L_base + λ [L, Ω]
    L_twisted = L_base + torsion_strength * commutator

    return L_twisted

  def _build_torsion_operator(self, winding_number):
    """
    Construit l'opérateur de torsion Ω_w

    Physique: Représente la rotation dans l'espace de configuration
    qui induit la frustration spectrale pour w ≥ 2
    """
    Omega = np.zeros((self.nodes, self.nodes))

    for i in range(self.nodes):
      r_i = self.positions[i] - self.center

      for j in range(self.nodes):
        r_j = self.positions[j] - self.center

        # Distance entre nœuds
        r_ij = np.linalg.norm(self.positions[i] - self.positions[j])

        if r_ij < 2.0:  # Couplage local uniquement
          # Vecteur reliant i à j
          dr = self.positions[j] - self.positions[i]

          # Composante azimutale (rotation autour de z)
          theta_i = np.arctan2(r_i[1], r_i[0])
          theta_j = np.arctan2(r_j[1], r_j[0])

          # Phase de Berry-like pour le winding
          # Pour w=1 : phase 2π
          # Pour w=2 : phase 4π (frustration critique)
          # Pour w=3 : phase 6π
          phase = winding_number * (theta_j - theta_i)

          # Couplage de torsion avec décroissance radiale
          r_i_norm = np.linalg.norm(r_i)
          r_j_norm = np.linalg.norm(r_j)

          profile = np.exp(-(r_i_norm ** 2 + r_j_norm ** 2) / (2 * self.chi_c ** 2))

          Omega[i, j] = profile * np.sin(phase) / (1 + r_ij)

    return csr_matrix(Omega)

  def _compute_background_field(self):
    """Champ de fond χ̄ par moyenne locale"""
    d_comb = squareform(pdist(self.positions, metric='euclidean'))
    cutoff = 1.5

    chi_bar = np.zeros(self.nodes)
    for i in range(self.nodes):
      neighbors = d_comb[i] <= cutoff
      chi_bar[i] = np.mean(self.chi[neighbors])

    return chi_bar

  def _build_coupling_matrix(self, chi_bar):
    """K_ij selon Eq. 12"""
    delta_chi = np.abs(chi_bar[:, None] - chi_bar[None, :])
    K = self.K0 * np.exp(-(delta_chi ** 2) / (self.chi_c ** 2))

    d_comb = squareform(pdist(self.positions, metric='euclidean'))
    K[d_comb > 1.5] = 0

    return csr_matrix(K)

  def compute_spectrum(self, laplacian, n_eigenvalues=20):
    """Calcule les valeurs propres"""
    eigenvalues, eigenvectors = eigsh(laplacian, k=n_eigenvalues, which='SM')
    idx = np.argsort(eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx]


def test1_torsion_operator():
  """
  TEST #1: Vérifier si Ω_w induit le ratio 8/3 pour w=2
  """
  print("\n" + "=" * 80)
  print(" TEST #1: EXPLICIT TORSION OPERATOR Ω_w ".center(80))
  print("=" * 80)
  print("\nHypothesis: λ₂/λ₁ = 8/3 emerges from non-commutativity [∆, Ω_w] ≠ 0")
  print("\nPrediction according to Cosmochrony:")
  print("  • w=0 (no torsion):  λ₂/λ₁ ~ 1.0 (baseline)")
  print("  • w=1 (electron):    λ₂/λ₁ ~ arbitrary")
  print("  • w=2 (muon):        λ₂/λ₁ ≈ 8/3 = 2.667 ← CRITICAL TEST")
  print("  • w=3 (proton):      λ₂/λ₁ > 8/3")
  print("\n" + "-" * 80)

  network = CosmoChronyNetworkWithTorsion(N=343, chi_c=1.0, K0=1.0)  # 7³ = 343

  results = {}
  winding_numbers = [0, 1, 2, 3]

  for w in winding_numbers:
    print(f"\n  → Winding number w = {w}")

    # Test différentes forces de torsion
    torsion_strengths = [0.1, 0.3, 0.5, 0.7, 1.0]

    for strength in torsion_strengths:
      L = network.build_laplacian_with_torsion(
        winding_number=w,
        torsion_strength=strength
      )

      eigenvalues, _ = network.compute_spectrum(L)

      lambda_1 = eigenvalues[1]
      lambda_2 = eigenvalues[2]
      ratio = lambda_2 / lambda_1

      deviation = abs(ratio - 8 / 3)

      if w not in results:
        results[w] = []
      results[w].append({
        'strength': strength,
        'ratio': ratio,
        'deviation': deviation
      })

      print(f"     Torsion strength = {strength:.1f}:  λ₂/λ₁ = {ratio:.6f}  "
            f"(dev = {deviation:.6f})", end="")

      if deviation < 0.1:
        print("  ✅ EXCELLENT")
      elif deviation < 0.5:
        print("  ✓ Good")
      else:
        print("")

  # Visualisation
  fig, axes = plt.subplots(1, 2, figsize=(16, 6))

  # Plot 1: Ratio vs. torsion strength pour chaque w
  ax = axes[0]
  for w in winding_numbers:
    strengths = [r['strength'] for r in results[w]]
    ratios = [r['ratio'] for r in results[w]]
    ax.plot(strengths, ratios, 'o-', linewidth=2, markersize=8, label=f'w = {w}')

  ax.axhline(y=8 / 3, color='red', linestyle='--', linewidth=2, label='Target: 8/3')
  ax.set_xlabel('Torsion strength', fontsize=12)
  ax.set_ylabel('λ₂/λ₁', fontsize=12)
  ax.set_title('Spectral Ratio vs. Torsion Strength', fontsize=14, fontweight='bold')
  ax.legend()
  ax.grid(True, alpha=0.3)

  # Plot 2: Deviation from 8/3
  ax = axes[1]
  for w in winding_numbers:
    strengths = [r['strength'] for r in results[w]]
    deviations = [r['deviation'] for r in results[w]]
    ax.semilogy(strengths, deviations, 'o-', linewidth=2, markersize=8, label=f'w = {w}')

  ax.axhline(y=0.1, color='green', linestyle='--', label='10% tolerance')
  ax.set_xlabel('Torsion strength', fontsize=12)
  ax.set_ylabel('|λ₂/λ₁ - 8/3|', fontsize=12)
  ax.set_title('Deviation from Target Ratio', fontsize=14, fontweight='bold')
  ax.legend()
  ax.grid(True, alpha=0.3)

  plt.tight_layout()
  plt.savefig('test1_torsion_operator.png', dpi=150, bbox_inches='tight')
  print("\n  Figure saved: test1_torsion_operator.png")

  # Verdict
  print("\n" + "-" * 80)
  print(" VERDICT TEST #1:")

  # Chercher le meilleur résultat pour w=2
  w2_results = results[2]
  best_w2 = min(w2_results, key=lambda x: x['deviation'])

  print(f"  Best result for w=2: λ₂/λ₁ = {best_w2['ratio']:.6f} "
        f"at torsion strength = {best_w2['strength']:.1f}")
  print(f"  Deviation from 8/3: {best_w2['deviation']:.6f}")

  if best_w2['deviation'] < 0.1:
    print("  ✅ HYPOTHESIS CONFIRMED: Torsion operator generates 8/3 ratio")
  elif best_w2['deviation'] < 0.5:
    print("  ⚠️  PARTIAL SUPPORT: Ratio closer to 8/3 but not exact")
  else:
    print("  ❌ HYPOTHESIS REJECTED: Torsion does not generate 8/3")

  return results


# ============================================================================
# TEST #2: S³ GEOMETRY
# ============================================================================

class CosmoChronyNetworkS3:
  """
  Réseau construit directement sur la 3-sphère S³ ⊂ R⁴
  avec distance géodésique et fibration de Hopf
  """

  def __init__(self, N=1000, chi_c=1.0, K0=1.0):
    self.nodes = N
    self.chi_c = chi_c
    self.K0 = K0

    # Génération de points uniformes sur S³
    self.positions = self._sample_S3(N)

    # Champ χ initial
    self.chi = np.random.randn(self.nodes) * 0.2

  def _sample_S3(self, N):
    """
    Échantillonnage uniforme sur S³ (Marsaglia 1972 généralisé)
    """
    points = []
    while len(points) < N:
      # 4 coordonnées gaussiennes
      x = np.random.randn(4)

      # Normalisation sur S³
      x_normalized = x / np.linalg.norm(x)
      points.append(x_normalized)

    return np.array(points)

  def geodesic_distance_matrix(self):
    """
    Calcule la matrice de distances géodésiques sur S³
    d(p, q) = arccos(p · q)
    """
    N = self.nodes
    dist = np.zeros((N, N))

    for i in range(N):
      for j in range(i + 1, N):
        # Produit scalaire dans R⁴
        dot_product = np.clip(np.dot(self.positions[i], self.positions[j]), -1.0, 1.0)

        # Distance géodésique
        d = np.arccos(dot_product)
        dist[i, j] = d
        dist[j, i] = d

    return dist

  def build_laplacian_S3(self):
    """
    Construit le Laplacien de graphe sur S³ avec couplage χ-dépendant
    """
    # Distance géodésique
    dist = self.geodesic_distance_matrix()

    # Champ de fond
    chi_bar = self._compute_background_field(dist)

    # Couplage constitutif
    delta_chi = np.abs(chi_bar[:, None] - chi_bar[None, :])
    K = self.K0 * np.exp(-(delta_chi ** 2) / (self.chi_c ** 2))

    # Cutoff géodésique (connectivité locale sur S³)
    # Sur S³, rayon π/4 ≈ 0.785 pour voisins proches
    cutoff = np.pi / 4
    K[dist > cutoff] = 0

    # Laplacien
    degree = np.sum(K, axis=1)
    D = diags(degree, 0)
    L = D - csr_matrix(K)

    return L

  def _compute_background_field(self, dist_matrix):
    """Champ de fond par moyenne géodésique locale"""
    cutoff = np.pi / 4
    chi_bar = np.zeros(self.nodes)

    for i in range(self.nodes):
      neighbors = dist_matrix[i] <= cutoff
      chi_bar[i] = np.mean(self.chi[neighbors])

    return chi_bar

  def compute_spectrum(self, n_eigenvalues=20):
    """Calcule spectre sur S³"""
    L = self.build_laplacian_S3()
    eigenvalues, eigenvectors = eigsh(L, k=n_eigenvalues, which='SM')
    idx = np.argsort(eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx]


def test2_S3_geometry():
  """
  TEST #2: Vérifier si la géométrie S³ induit le ratio 8/3
  """
  print("\n" + "=" * 80)
  print(" TEST #2: S³ GEOMETRY (Hopf Fibration) ".center(80))
  print("=" * 80)
  print("\nHypothesis: λ₂/λ₁ = 8/3 is a geometric property of S³, not flat R³")
  print("\nBackground: Hopf fibration S³ → S² has known spectral properties")
  print("  • S³ has symmetry group SO(4)")
  print("  • Laplacian spectrum on S³: λ_n = n(n+2) for harmonics")
  print("  • Ratio λ₂/λ₁ for pure S³ = 8/2 = 4 (not 8/3)")
  print("\nBUT: With χ-field coupling, spectrum may be modified...")
  print("\n" + "-" * 80)

  # Test différentes tailles
  sizes = [200, 500, 1000]

  results_S3 = []

  for N in sizes:
    print(f"\n  → S³ network with N = {N} points")

    network_S3 = CosmoChronyNetworkS3(N=N, chi_c=1.0, K0=1.0)
    eigenvalues, _ = network_S3.compute_spectrum()

    lambda_1 = eigenvalues[1]
    lambda_2 = eigenvalues[2]
    ratio = lambda_2 / lambda_1

    deviation = abs(ratio - 8 / 3)

    results_S3.append({
      'N': N,
      'ratio': ratio,
      'deviation': deviation,
      'eigenvalues': eigenvalues
    })

    print(f"     λ₁ = {lambda_1:.6e}")
    print(f"     λ₂/λ₁ = {ratio:.6f}  (target: 8/3 = 2.667)")
    print(f"     Deviation: {deviation:.6f}", end="")

    if deviation < 0.1:
      print("  ✅ EXCELLENT")
    elif deviation < 0.5:
      print("  ✓ Good")
    else:
      print("")

  # Visualisation
  fig, axes = plt.subplots(1, 2, figsize=(16, 6))

  # Plot 1: Convergence du ratio avec N
  ax = axes[0]
  Ns = [r['N'] for r in results_S3]
  ratios = [r['ratio'] for r in results_S3]

  ax.plot(Ns, ratios, 'o-', linewidth=3, markersize=12, color='blue', label='S³ network')
  ax.axhline(y=8 / 3, color='red', linestyle='--', linewidth=2, label='Target: 8/3')
  ax.axhline(y=4.0, color='orange', linestyle=':', linewidth=2, label='Pure S³: 4')
  ax.set_xlabel('Number of points N', fontsize=12)
  ax.set_ylabel('λ₂/λ₁', fontsize=12)
  ax.set_title('Spectral Ratio on S³ Geometry', fontsize=14, fontweight='bold')
  ax.legend()
  ax.grid(True, alpha=0.3)

  # Plot 2: Spectre complet pour N_max
  ax = axes[1]
  eigenvalues_max = results_S3[-1]['eigenvalues']
  ax.plot(eigenvalues_max, 'o-', linewidth=2, markersize=8)
  ax.set_xlabel('Mode index n', fontsize=12)
  ax.set_ylabel('Eigenvalue λₙ', fontsize=12)
  ax.set_title(f'Full Spectrum on S³ (N={sizes[-1]})', fontsize=14, fontweight='bold')
  ax.grid(True, alpha=0.3)

  plt.tight_layout()
  plt.savefig('test2_S3_geometry.png', dpi=150, bbox_inches='tight')
  print("\n  Figure saved: test2_S3_geometry.png")

  # Verdict
  print("\n" + "-" * 80)
  print(" VERDICT TEST #2:")

  best_result = min(results_S3, key=lambda x: x['deviation'])
  print(f"  Best result: λ₂/λ₁ = {best_result['ratio']:.6f} at N = {best_result['N']}")
  print(f"  Deviation from 8/3: {best_result['deviation']:.6f}")

  if best_result['deviation'] < 0.1:
    print("  ✅ HYPOTHESIS CONFIRMED: S³ geometry generates 8/3 ratio")
  elif best_result['deviation'] < 0.5:
    print("  ⚠️  PARTIAL SUPPORT: Ratio closer to 8/3 on S³ than on R³")
  else:
    print("  ❌ HYPOTHESIS REJECTED: S³ alone does not generate 8/3")

  return results_S3


# ============================================================================
# TEST #3: POTENTIAL V(χ) CORRECTIONS
# ============================================================================

class CosmoChronyNetworkWithPotential:
  """
  Réseau avec potentiel effectif V(χ) incluant terme de stabilisation
  """

  def __init__(self, N=512, chi_c=1.0, K0=1.0, lambda_potential=0.25):
    n = int(np.cbrt(N))
    self.nodes = n ** 3
    self.chi_c = chi_c
    self.K0 = K0
    self.lambda_potential = lambda_potential

    # Réseau cubique 3D
    x, y, z = np.meshgrid(np.arange(n), np.arange(n), np.arange(n))
    self.positions = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    # Champ χ initial avec structure
    self.chi = self._initialize_chi_with_structure()

  def _initialize_chi_with_structure(self):
    """
    Initialise χ avec configuration proche du minimum de V(χ)
    """
    # Potentiel double-puits: V(χ) = λ(χ² - χ_c²)²/4
    # Minima à χ = ±χ_c

    # Domaine avec χ ≈ +χ_c et χ ≈ -χ_c
    chi = np.random.choice([self.chi_c, -self.chi_c], size=self.nodes)

    # Ajouter fluctuations
    chi += np.random.randn(self.nodes) * 0.1 * self.chi_c

    return chi

  def V(self, chi):
    """Potentiel double-puits"""
    return self.lambda_potential * (chi ** 2 - self.chi_c ** 2) ** 2 / 4

  def dV_dchi(self, chi):
    """Gradient du potentiel"""
    return self.lambda_potential * chi * (chi ** 2 - self.chi_c ** 2)

  def build_laplacian_with_potential(self):
    """
    Construit Laplacien avec corrections de V(χ)

    K_ij modifié pour inclure stabilisation par V(χ)
    """
    # Distance combinatoriale
    d_comb = squareform(pdist(self.positions, metric='euclidean'))

    # Champ de fond
    chi_bar = self._compute_background_field(d_comb)

    # Couplage de base
    delta_chi = np.abs(chi_bar[:, None] - chi_bar[None, :])
    K_base = self.K0 * np.exp(-(delta_chi ** 2) / (self.chi_c ** 2))

    # Correction de V(χ) : modifie le couplage local
    # Intuition: Régions où V'(χ) est grand → couplage réduit (stabilisation)
    dV = self.dV_dchi(chi_bar)

    # Facteur de suppression basé sur le gradient de V
    # Plus |dV/dχ| est grand, plus le couplage est réduit
    suppression = np.exp(-np.abs(dV[:, None] + dV[None, :]) / (2 * self.chi_c))

    K_modified = K_base * suppression

    # Cutoff spatial
    cutoff = 1.5
    K_modified[d_comb > cutoff] = 0

    # Laplacien
    degree = np.sum(K_modified, axis=1)
    D = diags(degree, 0)
    L = D - csr_matrix(K_modified)

    return L

  def _compute_background_field(self, dist_matrix):
    """Champ de fond"""
    cutoff = 1.5
    chi_bar = np.zeros(self.nodes)

    for i in range(self.nodes):
      neighbors = dist_matrix[i] <= cutoff
      chi_bar[i] = np.mean(self.chi[neighbors])

    return chi_bar

  def compute_spectrum(self, n_eigenvalues=20):
    """Calcule spectre"""
    L = self.build_laplacian_with_potential()
    eigenvalues, eigenvectors = eigsh(L, k=n_eigenvalues, which='SM')
    idx = np.argsort(eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx]


def test3_potential_corrections():
  """
  TEST #3: Vérifier si V(χ) fournit corrections fines vers 8/3
  """
  print("\n" + "=" * 80)
  print(" TEST #3: POTENTIAL V(χ) CORRECTIONS ".center(80))
  print("=" * 80)
  print("\nHypothesis: V(χ) provides fine corrections shifting λ₂/λ₁ from ~1.0 to ~2.67")
  print("\nDocument claim (Section B.10):")
  print('  "V(χ) is expected to control fine splittings within a given solitonic sector"')
  print("\nTest: Double-well potential V(χ) = λ(χ² - χ_c²)²/4")
  print("\n" + "-" * 80)

  # Test différentes forces du potentiel
  lambda_values = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]

  results_potential = []

  for lambda_pot in lambda_values:
    print(f"\n  → Potential strength λ = {lambda_pot:.2f}")

    network = CosmoChronyNetworkWithPotential(
      N=343,  # 7³
      chi_c=1.0,
      K0=1.0,
      lambda_potential=lambda_pot
    )

    eigenvalues, _ = network.compute_spectrum()

    lambda_1 = eigenvalues[1]
    lambda_2 = eigenvalues[2]
    ratio = lambda_2 / lambda_1

    deviation = abs(ratio - 8 / 3)

    results_potential.append({
      'lambda': lambda_pot,
      'ratio': ratio,
      'deviation': deviation,
      'eigenvalues': eigenvalues
    })

    print(f"     λ₁ = {lambda_1:.6e}")
    print(f"     λ₂/λ₁ = {ratio:.6f}  (target: 8/3 = 2.667)")
    print(f"     Deviation: {deviation:.6f}", end="")

    if deviation < 0.1:
      print("  ✅ EXCELLENT")
    elif deviation < 0.5:
      print("  ✓ Good")
    else:
      print("")

  # Visualisation
  fig, axes = plt.subplots(1, 2, figsize=(16, 6))

  # Plot 1: Ratio vs. potential strength
  ax = axes[0]
  lambdas = [r['lambda'] for r in results_potential]
  ratios = [r['ratio'] for r in results_potential]

  ax.plot(lambdas, ratios, 'o-', linewidth=3, markersize=12, color='purple')
  ax.axhline(y=8 / 3, color='red', linestyle='--', linewidth=2, label='Target: 8/3')
  ax.set_xlabel('Potential strength λ', fontsize=12)
  ax.set_ylabel('λ₂/λ₁', fontsize=12)
  ax.set_title('Spectral Ratio vs. Potential Strength', fontsize=14, fontweight='bold')
  ax.legend()
  ax.grid(True, alpha=0.3)

  # Plot 2: Spectre pour différentes forces
  ax = axes[1]
  for i, result in enumerate(results_potential[::2]):  # Every other result
    eigenvalues = result['eigenvalues']
    ax.plot(eigenvalues, 'o-', linewidth=2, markersize=6,
            label=f"λ = {result['lambda']:.2f}")

  ax.set_xlabel('Mode index n', fontsize=12)
  ax.set_ylabel('Eigenvalue λₙ', fontsize=12)
  ax.set_title('Spectrum Evolution with V(χ)',fontsize=14, fontweight='bold')
  ax.legend()
  ax.grid(True, alpha=0.3)

  plt.tight_layout()
  plt.savefig('test3_potential_corrections.png', dpi=150, bbox_inches='tight')
  print("\n  Figure saved: test3_potential_corrections.png")

  # Verdict
  print("\n" + "-" * 80)
  print(" VERDICT TEST #3:")

  best_result = min(results_potential, key=lambda x: x['deviation'])
  print(f"  Best result: λ₂/λ₁ = {best_result['ratio']:.6f} at λ = {best_result['lambda']:.2f}")
  print(f"  Deviation from 8/3: {best_result['deviation']:.6f}")

  # Comparer avec baseline (λ=0)
  baseline = results_potential[0]
  improvement = baseline['deviation'] - best_result['deviation']

  print(f"  Improvement over baseline: {improvement:.6f}")

  if best_result['deviation'] < 0.1:
    print("  ✅ HYPOTHESIS CONFIRMED: V(χ) generates 8/3 ratio")
  elif improvement > 0.5:
    print("  ⚠️  PARTIAL SUPPORT: V(χ) improves ratio significantly")
  else:
    print("  ❌ HYPOTHESIS REJECTED: V(χ) provides minimal correction")

  return results_potential

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
  print("\nStarting comprehensive critical tests...")
  print("This may take several minutes depending on network sizes.\n")
  # Execute all three tests
  results1 = test1_torsion_operator()
  results2 = test2_S3_geometry()
  results3 = test3_potential_corrections()

  # Final summary
  print("\n" + "=" * 80)
  print(" FINAL SUMMARY: Critical Assessment of λ₂/λ₁ = 8/3 Prediction ".center(80))
  print("=" * 80)

  # Find best results from each test
  best_w2_torsion = min(results1[2], key=lambda x: x['deviation'])
  best_S3 = min(results2, key=lambda x: x['deviation'])
  best_potential = min(results3, key=lambda x: x['deviation'])

  print("\nBest deviations from target (8/3 = 2.667):")
  print(f"  [1] Torsion operator (w=2):  {best_w2_torsion['deviation']:.6f} "
        f"(strength={best_w2_torsion['strength']:.1f})")
  print(f"  [2] S³ geometry:              {best_S3['deviation']:.6f} "
        f"(N={best_S3['N']})")
  print(f"  [3] Potential V(χ):           {best_potential['deviation']:.6f} "
        f"(λ={best_potential['lambda']:.2f})")

  print("\n" + "-" * 80)
  print("OVERALL ASSESSMENT:")

  min_deviation = min(
    best_w2_torsion['deviation'],
    best_S3['deviation'],
    best_potential['deviation']
  )

  if min_deviation < 0.1:
    print("  ✅ SUCCESS: At least one mechanism generates λ₂/λ₁ ≈ 8/3")
    print("     → Cosmochrony prediction is VALIDATED")
    print("     → Mechanism identified for inclusion in next paper version")
  elif min_deviation < 0.5:
    print("  ⚠️  PARTIAL SUCCESS: Ratio improved but not exact")
    print("     → Combination of mechanisms may be required")
    print("     → Further refinement needed")
  else:
    print("  ❌ FAILURE: None of the tested mechanisms generate 8/3")
    print("     → λ₂/λ₁ = 8/3 appears to be phenomenological input, not output")
    print("     → Paper should revise claim from 'derived' to 'constrained'")

  print("\n" + "=" * 80)
  print(" RECOMMENDATION FOR PAPER:")

  if min_deviation < 0.3:
    print("  • Keep λ₂/λ₁ = 8/3 as theoretical prediction")
    print("  • Add section explaining mechanism (torsion/S³/potential)")
    print("  • Provide numerical validation in supplementary material")
  else:
    print("  • Reframe λ₂/λ₁ = 8/3 as phenomenological constraint")
    print("  • Be explicit: 'This ratio requires further derivation'")
    print("  • List as open problem in 'Future Work' section")

  print("=" * 80 + "\n")