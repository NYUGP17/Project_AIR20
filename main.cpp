#include <igl/readOFF.h>
#include <igl/per_vertex_normals.h>
#include <igl/per_face_normals.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/bounding_box_diagonal.h>
#include <igl/doublearea.h>
#include <igl/viewer/Viewer.h>

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
// marks singular triangles (#F * 1)
// singular = 1, regular = 0
VectorXi singular_indices(0, 1);

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
			h = std::sqrt((1.0 + n1.dot(n2)/2.0));
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
		Kmax[v] = solver.eigenvalues()[indices[0]];
		Kmin[v] = solver.eigenvalues()[indices[1]];
		Kdmax.row(v) = solver.eigenvectors().col(indices[0]).normalized();
		Kdmin.row(v) = solver.eigenvectors().col(indices[1]).normalized();
	}
}

void mark_singular_triangles(
	const MatrixXi &F,
	const MatrixXd &Kdmax,
	const MatrixXd &Kdmin,
	VectorXi &singular_indices)
{
	singular_indices.resize(F.rows(), 1);
	for (int f = 0; f < F.rows(); f++) {
		double sign_min = 1.0;
		double sign_max = 1.0;
		for (int i = 0; i < 3; i++)
			for (int j = i+1; j < 3; j++) {
				sign_min *= Kdmin.row(F(f, i)).dot(Kdmin.row(F(f, j)));
				sign_max *= Kdmax.row(F(f, i)).dot(Kdmax.row(F(f, j)));
		}
		if (sign_min > 0.0 && sign_max > 0.0) {
			singular_indices[f] = 0;
		} else {
			singular_indices[f] = 1;
		}
	}
	
}

void show_curvatures(Viewer &viewer)
{
	double coeff = 0.01 * igl::bounding_box_diagonal(V);
	viewer.data.clear();
	viewer.data.add_edges(V, V + coeff * Kdmin, RowVector3d(0.0, 1.0, 0.0));
	viewer.data.add_edges(V, V - coeff * Kdmin, RowVector3d(0.0, 1.0, 0.0));
	viewer.data.add_edges(V, V + coeff * Kdmax, RowVector3d(1.0, 0.0, 0.0));
	viewer.data.add_edges(V, V - coeff * Kdmax, RowVector3d(1.0, 0.0, 0.0));
}

void show_singular_triangles(Viewer &viewer)
{
	const RowVector3d white = RowVector3d(1.0, 1.0, 1.0);
	const RowVector3d red = RowVector3d(1.0, 0.0, 0.0);
	MatrixXd face_colors(F.rows(), 3);
	for (int f = 0; f < F.rows(); f++) {
		if (singular_indices[f] == 0) {
			face_colors.row(f) = white;
		} else {
			face_colors.row(f) = red;
		}
	}
	viewer.data.clear();
	viewer.data.set_mesh(V, F);
	viewer.data.set_colors(face_colors);
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
	mark_singular_triangles(F, Kdmax, Kdmin, singular_indices);

	igl::viewer::Viewer viewer;
	viewer.data.set_mesh(V, F);

	// Configure nanagui sidebar menu
	viewer.callback_init = [&](Viewer &v) {
		v.ngui->addButton("Show curvatures", [&](){
			show_curvatures(v);
		});
		v.ngui->addButton("Show singular triangles", [&](){
			show_singular_triangles(v);
		});
		v.screen->performLayout();
		return false;
	};
	viewer.launch();

	return 0;
}
