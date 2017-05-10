# Geometric processing class project

Extract smooth feature lines given an input mesh.

## Compile

Compile this project using the standard cmake routine:

    mkdir build
    cd build
    cmake ..
    make

This should find and build the dependencies and create a `ftlext` binary.

## Dependencies

The only dependencies are stl, eigen, [libigl](libigl.github.io/libigl/) and
the dependencies of the `igl::viewer::Viewer` (mandatory: glfw and
opengl, optional: nanogui and nanovg).

We recommend you to install libigl using git via:

    git clone --recursive https://github.com/libigl/libigl.git

If you have installed libigl at `/path/to/libigl/` then a good place to clone
this library is `/path/to/libigl-example-project/`.
