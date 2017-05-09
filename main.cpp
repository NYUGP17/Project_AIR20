#include <igl/readOFF.h>
#include <igl/viewer/Viewer.h>

using namespace Eigen;

MatrixXd V(0, 3);
MatrixXi F(0, 3);

int main(int argc, char *argv[])
{
	if (argc != 2) {
		std::cout << "Usage ftlext mesh.off" << std::endl;
		exit(0);
	}
	
	// Read mesh data
	igl::readOFF(argv[1], V, F);
	igl::viewer::Viewer viewer;
	viewer.data.set_mesh(V, F);
	viewer.data.set_face_based(true);

	// Configure nanagui sidebar menu
	viewer.callback_init = [&](igl::viewer::Viewer &v) {
		v.screen->performLayout();
		return false;
	};
	viewer.launch();

	return 0;
}
