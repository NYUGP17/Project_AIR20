#include <igl/readOFF.h>
#include <igl/per_vertex_normals.h>
#include <igl/per_face_normals.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/triangle_triangle_adjacency.h>
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

// compute curvatures and principal directions
void principal_curvatures(
	const MatrixXd &V,
	const MatrixXi &F,
	VectorXd &Kmin,
	VectorXd &Kmax,
	MatrixXd &Kdmin,
	MatrixXd &Kdmax)
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
			assert(f2 >= 0); // assume no boundary edges
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
		Kmin[v] = solver.eigenvalues()[1];
		Kmax[v] = solver.eigenvalues()[2];
		Kdmin.row(v) = solver.eigenvectors().col(1).normalized();
		Kdmax.row(v) = solver.eigenvectors().col(2).normalized();
	}
}

void show_curvatures(Viewer &viewer)
{
	viewer.data.clear();
	viewer.data.add_edges(V, V + 0.1 * Kdmin, RowVector3d(0.0, 1.0, 0.0));
	viewer.data.add_edges(V, V - 0.1 * Kdmin, RowVector3d(0.0, 1.0, 0.0));
	viewer.data.add_edges(V, V + 0.1 * Kdmax, RowVector3d(1.0, 0.0, 0.0));
	viewer.data.add_edges(V, V - 0.1 * Kdmax, RowVector3d(1.0, 0.0, 0.0));
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
	principal_curvatures(V, F, Kmin, Kmax, Kdmin, Kdmax);

	igl::viewer::Viewer viewer;
	viewer.data.set_mesh(V, F);

	// Configure nanagui sidebar menu
	viewer.callback_init = [&](Viewer &v) {
		v.ngui->addButton("Show curvatures", [&](){
			show_curvatures(v);
		});
		v.screen->performLayout();
		return false;
	};
	viewer.launch();

	return 0;
}
