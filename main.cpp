#include <igl/readOFF.h>
#include <igl/per_vertex_normals.h>
#include <igl/per_face_normals.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/bounding_box_diagonal.h>
#include <igl/doublearea.h>
#include <igl/grad.h>
#include <igl/slice.h>
#include <igl/jet.h>
#include <igl/adjacency_list.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <igl/viewer/Viewer.h>
#include <set>

using namespace Eigen;
using Viewer = igl::viewer::Viewer;

// vertices and faces
MatrixXd V(0, 3);
MatrixXi F(0, 3);

// per vertex principal curvatures (#V * 1)
VectorXd Kmin(0, 1);
VectorXd Kmax(0, 1);
// per vertex principal directions (#V * 3)
MatrixXd Kdmin(0, 3);
MatrixXd Kdmax(0, 3);
// list of indices that are singular
VectorXi singular_indices(0, 1);
// list of indices that are regular
VectorXi regular_indices(0, 1);
// per vertex extremalities
VectorXd Exmin(0, 1);
VectorXd Exmax(0, 1);

int sgn(double x) { return (x > 0) - (x < 0); }

// compute curvatures and principal directions
void principal_curvatures(
	const MatrixXd &V,
	const MatrixXi &F,
	VectorXd &Kmax,
	VectorXd &Kmin,
	MatrixXd &Kdmax,
	MatrixXd &Kdmin)
{
	MatrixXd N, Nf; // per vertex and face normals
	igl::per_vertex_normals(V, F, N);
	igl::per_face_normals(V, F, Nf);

	std::vector<std::vector<int> > VF, VFi; // vertex-face adjacency list
	MatrixXi TT, TTi; // face-face adjacency list
	igl::vertex_triangle_adjacency(V, F, VF, VFi);
	igl::triangle_triangle_adjacency(F, TT, TTi);

	VectorXd A; // double area
	igl::doublearea(V, F, A);

	// initialize result matrices
	Kmin.resize(V.rows());
	Kmax.resize(V.rows());
	Kdmin.resize(V.rows(), 3);
	Kdmax.resize(V.rows(), 3);

	for (int v = 0; v < V.rows(); v++) {
		// construct the 3x3 matrix
		Matrix3d S = Matrix3d::Zero();
		for (int i = 0; i < VF[v].size(); i++) {
			int f1 = VF[v][i];
			int f2 = TT(f1, VFi[v][i]);
			if (f2 < 0) continue; // skip boundary edges
			int vn = F(f1, (VFi[v][i] + 1) % 3);
			Vector3d e = V.row(v) - V.row(vn); // edge vector
			Vector3d ne = (N.row(v) + N.row(vn)).normalized(); // edge normal
			double h; // cos half angle between faces
			Vector3d n1 = Nf.row(f1);
			Vector3d n2 = Nf.row(f2);
			h = 2.0 * e.norm() * std::sqrt((1.0 - n1.dot(n2))/2.0);
			double coeff = 0.5 * ne.dot(N.row(v));
			S += coeff * h * e.cross(ne) * e.cross(ne).transpose();
		}
		SelfAdjointEigenSolver<Matrix3d> solver(S);
		// find the largest two absolute values
		std::vector<int> indices = {0, 1, 2};
		std::sort(indices.begin(), indices.end(), [&](int a, int b){
			return std::abs(solver.eigenvalues()[a]) > std::abs(solver.eigenvalues()[b]);
		});
		indices.resize(2);
		std::sort(indices.begin(), indices.end(), [&](int a, int b){
			return solver.eigenvalues()[a] > solver.eigenvalues()[b];
		});
		// scale by the area
		double area = 0.0;
		for (int i = 0; i < VF[v].size(); i++)
			area += A[VF[v][i]];
		Kmax[v] = 1e3 * solver.eigenvalues()[indices[0]] / area;
		Kmin[v] = 1e3 * solver.eigenvalues()[indices[1]] / area;
		Kdmax.row(v) = solver.eigenvectors().col(indices[0]).normalized();
		Kdmin.row(v) = solver.eigenvectors().col(indices[1]).normalized();
	}
}

void extremality_coefficients(
	const MatrixXd &V,
	const MatrixXi &F,
	const VectorXd &K,
	const MatrixXd &Kd,
	const VectorXi &regular_indices,
	VectorXd &Ex)
{
	Ex.resize(V.rows());

	VectorXd A; // double area
	igl::doublearea(V, F, A);

	SparseMatrix<double> G; // gradient
	igl::grad(V, F, G);
	VectorXd gK = VectorXd(G * K);

	std::vector<std::vector<int> > VF, VFi; // vertex-face adjacency list
	MatrixXi TT, TTi; // face-face adjacency list
	igl::vertex_triangle_adjacency(V, F, VF, VFi);
	igl::triangle_triangle_adjacency(F, TT, TTi);

	for (int v = 0; v < V.rows(); v++) {
		double area = 0.0;
		double ex = 0.0;

		for (int i = 0; i < VF[v].size(); i++)
			area += A[VF[v][i]];
		for (int i = 0; i < VF[v].size(); i++) {
			int f = VF[v][i];
			Vector3d g = Vector3d(
				gK[0 * F.rows() + f],
				gK[1 * F.rows() + f],
				gK[2 * F.rows() + f]
			);
			ex += A[f] * g.dot(Kd.row(v));
		}
		ex /= area;
		Ex[v] = ex;
	}
}

void mark_singular_triangles(
	const MatrixXi &F,
	const MatrixXd &Kdmax,
	const MatrixXd &Kdmin,
	VectorXi &singular_indices,
	VectorXi &regular_indices)
{
	std::vector<int> singular_index_vector;
	std::vector<int> regular_index_vector;
	for (int f = 0; f < F.rows(); f++) {
		double sign_min = 1.0;
		double sign_max = 1.0;
		for (int i = 0; i < 3; i++)
			for (int j = i+1; j < 3; j++) {
				sign_min *= Kdmin.row(F(f, i)).dot(Kdmin.row(F(f, j)));
				sign_max *= Kdmax.row(F(f, i)).dot(Kdmax.row(F(f, j)));
			}
		if (sign_min > 0.0 && sign_max > 0.0) {
			regular_index_vector.push_back(f);
		} else {
			singular_index_vector.push_back(f);
		}
	}
	regular_indices.resize(regular_index_vector.size());
	for (int i = 0; i < regular_index_vector.size(); i++)
		regular_indices[i] = regular_index_vector[i];
	singular_indices.resize(singular_index_vector.size());
	for (int i = 0; i < singular_index_vector.size(); i++)
		singular_indices[i] = singular_index_vector[i];
}

void extract_feature_line(
	const MatrixXd &V,
	const MatrixXi &F,
	const VectorXd &Kmax,
	const VectorXd &Kmin,
	const MatrixXd &Kd,
	const VectorXd &Ex,
	const VectorXi &regular_indices,
	double sign, // workaround emax -> +1, emin -> -1
	MatrixXd &edges_start,
	MatrixXd &edges_end)
{
	// only consider regular faces
	MatrixXi Fr;
	igl::slice(F, regular_indices, 1, Fr);

	std::vector<std::vector<int> > VF, VFi; // vertex-face adjacency list
	MatrixXi TT, TTi; // face-face adjacency list
	igl::vertex_triangle_adjacency(V, Fr, VF, VFi);
	igl::triangle_triangle_adjacency(Fr, TT, TTi);

	std::vector<Vector3d> edges[2];
	std::set<std::pair<int, int>> marked_edges;

	for (int f = 0; f < Fr.rows(); f++) {
		Vector3d ex;
		Matrix3d kd;
		igl::slice(Ex, Fr.row(f), 1, ex);
		igl::slice(Kd, Fr.row(f), 1, kd);
		// flip signs to be consistent
		for (int i = 1; i < 3; i++) {
			if (kd.row(0).dot(kd.row(i)) < 0) {
				kd.row(i) = -kd.row(i);
				ex[i] = -ex[i];
			}
		}

		if ((ex.array() == 0.0).any())
			continue; // skip zeros
		if (ex.maxCoeff() <= 0.0 || ex.minCoeff() >= 0.0)
			continue; // no zero points

		Vector3d kd_sum = kd.colwise().sum();
		SparseMatrix<double> G; // gradient
		MatrixXd vv;
		igl::slice(V, Fr.row(f), 1, vv);
		igl::grad(vv, MatrixXi(RowVector3i(0, 1, 2)), G);
		VectorXd gex = VectorXd(G * ex);
		if (sign * gex.dot(kd_sum) >= 0.0)
			continue; // condition 1 not satisfied

		VectorXd kmax, kmin;
		igl::slice(Kmax, Fr.row(f), 1, kmax);
		igl::slice(Kmin, Fr.row(f), 1, kmin);
		if (sign * (std::abs(kmax.colwise().sum()[0]) - std::abs(kmin.colwise().sum()[0])) <= 0.0)
			continue; // condition 2 not satisfied

		int count = 0;
		for (int i = 0; i < 3; i++)
			for (int j = i+1; j < 3; j++) {
				if (ex[i] * ex[j] < 0) {
					// add mid point
					double a = std::abs(ex[i]);
					double b = std::abs(ex[j]);
					Vector3d mid = (b*V.row(Fr(f, i)) + a*V.row(Fr(f, j))) / (a + b);
					edges[count].push_back(mid);
					// mark edge j i
					marked_edges.insert(std::make_pair(Fr(f, j), Fr(f, i)));
					count++;
				}
			}
		assert(count == 2);

	}
	// processing singular triangles
	for (int i = 0; i < singular_indices.size(); i++) {
		int fi = singular_indices[i];
		std::vector<std::pair<int, int>> marked;
		for (int es = 0; es < 3; es++)
			for (int ee = es+1; ee < 3; ee++) {
				auto search = marked_edges.find(std::make_pair(F(fi, es), F(fi, ee)));
				if (search != marked_edges.end())
					marked.push_back(*search);
			}
		if (marked.size() < 2)
			continue;
		if (marked.size() == 2) {
			// connect two marked edges
			for (int k = 0; k < 2; k++) {
				int i = marked[k].first;
				int j = marked[k].second;
				double a = std::abs(Ex[i]);
				double b = std::abs(Ex[j]);
				Vector3d mid = (b*V.row(i) + a*V.row(j)) / (a + b);
				edges[k].push_back(mid);
			}
		}
		if (marked.size() == 3) {
			Vector3d v0 = V.row(F(fi, 0));
			Vector3d v1 = V.row(F(fi, 1));
			Vector3d v2 = V.row(F(fi, 2));
			Vector3d bc = (v0 + v1 + v2) / 3.0; // barycenter
			// connect marked edges to the barycenter
			for (int k = 0; k < 3; k++)
				edges[0].push_back(bc);
			for (int k = 0; k < 3; k++) {
				int i = marked[k].first;
				int j = marked[k].second;
				double a = std::abs(Ex[i]);
				double b = std::abs(Ex[j]);
				Vector3d mid = (b*V.row(i) + a*V.row(j)) / (a + b);
				edges[1].push_back(mid);
			}

		}

	}
	// create start points and end points
	edges_start.resize(edges[0].size(), 3);
	edges_end.resize(edges[1].size(), 3);
	for (int i = 0; i < edges[0].size(); i++) {
		edges_start.row(i) = edges[0][i];
		edges_end.row(i) = edges[1][i];
	}
}

void smooth_extremality(
	const MatrixXd &V,
	const MatrixXi &F,
	const MatrixXd &Kd,
	VectorXd &Ex)
{
	std::vector<std::vector<int> > A; // adjacency list
	igl::adjacency_list(F, A);

	// calculate cotan laplacian
	SparseMatrix<double> L, M, Minv;
	igl::cotmatrix(V, F, L);
	igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, M);
	igl::invert_diag(M, Minv);
	L = Minv * L;

	VectorXd LEx(V.rows());
	for (int v = 0; v < V.rows(); v++) {
		double lp = 0.0;
		for (int vn : A[v]) {
			int sign = sgn(Kd.row(v).dot(Kd.row(vn)));
			lp += L.coeffRef(v, vn) * (sign * Ex[vn] - Ex[v]);
		}
		LEx[v] = lp;
	}
	Ex += LEx;
}

void show_curvatures(Viewer &viewer) {
	static int mode = 0;
	viewer.data.clear();
	viewer.data.set_mesh(V, F);
	MatrixXd color_map;
	if (mode == 1) {
		igl::jet(Kmin, true, color_map);
		viewer.data.set_colors(color_map);
	} else if (mode == 0) {
		igl::jet(Kmax, true, color_map);
		viewer.data.set_colors(color_map);
	}
	mode = 1 - mode;

}

void show_principal_directions(Viewer &viewer)
{
	static int mode = 0;
	double coeff = 0.01 * igl::bounding_box_diagonal(V);
	viewer.data.clear();
	if (mode == 1) {
		viewer.data.add_edges(V, V + coeff * Kdmin, RowVector3d(0.0, 1.0, 0.0));
		viewer.data.add_edges(V, V - coeff * Kdmin, RowVector3d(0.0, 1.0, 0.0));
	} else if(mode == 0) {
		viewer.data.add_edges(V, V + coeff * Kdmax, RowVector3d(1.0, 0.0, 0.0));
		viewer.data.add_edges(V, V - coeff * Kdmax, RowVector3d(1.0, 0.0, 0.0));
	}
	mode = 1 - mode;
}

void show_singular_triangles(Viewer &viewer)
{
	const RowVector3d white = RowVector3d(1.0, 1.0, 1.0);
	const RowVector3d red = RowVector3d(1.0, 0.0, 0.0);
	MatrixXd face_colors(F.rows(), 3);
	for (int i = 0; i < singular_indices.rows(); i++) {
		face_colors.row(singular_indices[i]) = red;
	}
	for (int i = 0; i < regular_indices.rows(); i++) {
		face_colors.row(regular_indices[i]) = white;
	}
	viewer.data.clear();
	viewer.data.set_mesh(V, F);
	viewer.data.set_colors(face_colors);
}

void show_extremalities(Viewer &viewer)
{
	static int mode = 0;
	Eigen::MatrixXd color_map;
	if (mode == 1)
		igl::jet(Exmin, true, color_map);
	else
		igl::jet(Exmax, true, color_map);

	MatrixXi Fr;
	igl::slice(F, regular_indices, 1, Fr);

	viewer.data.clear();
	viewer.data.set_mesh(V, Fr);
	viewer.data.set_colors(color_map);

	mode = 1 - mode;
}

void show_feature_lines(Viewer &viewer) {
	MatrixXd start[2], end[2];
	extract_feature_line(V, F, Kmax, Kmin, Kdmax, Exmax, regular_indices, +1.0, start[0], end[0]);
	extract_feature_line(V, F, Kmax, Kmin, Kdmin, Exmin, regular_indices, -1.0, start[1], end[1]);

	viewer.data.clear();
	viewer.data.set_mesh(V, F);
	viewer.data.add_edges(start[0], end[0], RowVector3d(1.0, 0.0, 0.0));
	viewer.data.add_edges(start[1], end[1], RowVector3d(0.0, 0.0, 1.0));

}

int main(int argc, char *argv[])
{
	if (argc != 2) {
		std::cout << "Usage ftlext mesh.off" << std::endl;
		exit(0);
	}
	
	// Read mesh data
	igl::readOFF(argv[1], V, F);

	// Extract feature lines
	principal_curvatures(V, F, Kmax, Kmin, Kdmax, Kdmin);
	mark_singular_triangles(F, Kdmax, Kdmin, singular_indices, regular_indices);
	extremality_coefficients(V, F, Kmax, Kdmax, regular_indices, Exmax);
	extremality_coefficients(V, F, Kmin, Kdmin, regular_indices, Exmin);

	igl::viewer::Viewer viewer;
	viewer.data.set_mesh(V, F);

	// Configure nanagui sidebar menu
	viewer.callback_init = [&](Viewer &v) {
		v.ngui->addButton("Show curvatures", [&](){
			show_curvatures(v);
		});
		v.ngui->addButton("Show principal directions", [&](){
			show_principal_directions(v);
		});
		v.ngui->addButton("Show singular triangles", [&](){
			show_singular_triangles(v);
		});
		v.ngui->addButton("Show extremalities", [&](){
			show_extremalities(v);
		});
		v.ngui->addButton("Smooth extremalities", [&](){
			smooth_extremality(V, F, Kmax, Exmax);
			smooth_extremality(V, F, Kmin, Exmin);
		});
		v.ngui->addButton("Show feature line", [&](){
			show_feature_lines(v);
		});
		v.screen->performLayout();
		return false;
	};
	viewer.launch();

	return 0;
}
