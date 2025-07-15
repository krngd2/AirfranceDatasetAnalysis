import airfrans as af
import pyvista as pv
import numpy as np
import os

PATH_TO_DATASET = "airfrans_dataset"

# Dataset is composed of 1000 dirs (1 for each simulation)
# Each dir has 3 files (.vtp and .vtu -> XML files  (.vtp + .vtu = Newer version of .vtk))
#       - aerofoil.vtp -> internal boundary 
#               (Contains information about the internal boundary (the airfoil surface).)
#       - freestream.vtp -> external boundary 
#               (Represents the external boundary of the computational domain. This is where the airflow enters or exits the simulation.)
#       - internal.vtu -> internal field 
#               (Contains the mesh and solution data (e.g., velocity, pressure, turbulence fields) for every point in the computational grid. Includes both the freestream region and the disturbed airflow around the airfoil.)
# Manifest file contains what dirs/simulations should be used for training for each task

def get_image_from_sim(simulation, var="nut",output_image="airfRANS_cropped.png", airfoil=False, resolution=(1500, 750)):
    
    # Check if the variable is valid
    valid_fields = ["nut", "p", "U", "Ux", "Uy"]
    if var not in valid_fields:
        raise ValueError(f"{var} field not valid. Choose from {valid_fields}")
    
    # Fetch simulation
    sim = af.Simulation(root=PATH_TO_DATASET, name=simulation)

    # internal_mesh = pv.read(PATH_TO_DATASET+"/Dataset/"+simulation+"/"+simulation + "_internal.vtu")  # Fluid domain
    # freestream = pv.read(PATH_TO_DATASET+"/Dataset/"+simulation+"/"+simulation + "_freestream.vtp")  # Freestream boundary
    # airfoil = pv.read(PATH_TO_DATASET+"/Dataset/"+simulation+"/"+simulation + "_aerofoil.vtp")  # Airfoil geometry

    # Load meshes using airfRANS (no manual file reading needed)
    internal_mesh = sim.internal  # Fluid domain
    # freestream = sim.freestream()  # Freestream boundary -> bib does not provied acess to this
    airfoil = sim.airfoil  # Airfoil surface

    full_bounds = [-2, 4, -1.5, 1.5, 0.5, 0.5] 

    # Apply cropping (keep only the area near the airfoil)
    cropped_mesh = internal_mesh.clip_box(full_bounds, invert=False)

    # Create a PyVista plotter
    plotter = pv.Plotter(off_screen=True)

    # Print available scalar fields
    # print("Available Scalars:", cropped_mesh.array_names)

    # Extract the correct scalar field
    if var in ["Ux", "Uy"]:
        if "U" not in cropped_mesh.array_names:
            raise ValueError("Velocity field 'U' not found in simulation data.")
        velocity_field = cropped_mesh["U"]  # Shape (num_nodes, 3) -> (Ux, Uy, Uz)

        # Extract individual velocity components
        if var == "Ux":
            scalar_field = velocity_field[:, 0]  # Extract X-component
        elif var == "Uy":
            scalar_field = velocity_field[:, 1]  # Extract Y-component
    else:
        scalar_field = cropped_mesh[var]  # Use scalar field directly

    # Add internal mesh with turbulent viscosity ("nut") as the scalar field
    plotter.add_mesh(cropped_mesh, scalars=scalar_field, cmap="gray", show_edges=False)

    if airfoil:
        # Add airfoil geometry in black
        plotter.add_mesh(airfoil, color="black", line_width=1)

    # Set Transparent Background
    plotter.background_color = None  # Transparent background

    # Remove color bar
    plotter.remove_scalar_bar()

    # Ensure Camera Captures Top-Down View
    plotter.view_xy()
    # Center camera on the mesh
    plotter.camera.focal_point = [np.mean(full_bounds[:2]),np.mean(full_bounds[2:4]),np.mean(full_bounds[4:])]
    # print(cropped_mesh.center)
    # Avoid perspective distortion
    plotter.camera.parallel_projection = True

    # Calculate Mesh Extents
    mesh_width = abs(full_bounds[1] - full_bounds[0])
    mesh_height = abs(full_bounds[3] - full_bounds[2])

    # Set Desired Output Resolution
    image_aspect_ratio = resolution[0] / resolution[1]
    mesh_aspect_ratio = mesh_width / mesh_height

    # Adjust Camera Zoom to Fit Mesh Exactly
    if image_aspect_ratio > mesh_aspect_ratio:
        # Image is wider than mesh, fit by height
        plotter.camera.SetParallelScale(mesh_height / 2)
    else:
        # Image is taller than mesh, fit by width (adjust using aspect ratio)
        plotter.camera.SetParallelScale((mesh_width / 2) / image_aspect_ratio)
    # if output_image folder does not exist, create it

    output_dir = os.path.dirname(output_image)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Save and show the image
    # plotter.show(screenshot=output_image, window_size=resolution)
    # only save the image
    plotter.screenshot(output_image, window_size=resolution)
    # return cropped_mesh

def create_dataset():
    # Load test dataset list
    test_dataset_list, test_dataset_names = af.dataset.load(root=PATH_TO_DATASET+"/Dataset", task='full', train=False)

    # Save initial conditions for test dataset
    # save_initial_conditions(test_dataset_names, output="image_af/test.json")

    for idx,sim in enumerate(test_dataset_names): 
        get_image_from_sim(sim, var="nut",output_image="image_af/viscosity/test/"+sim+".png", airfoil=False, resolution=(1500, 750))
        get_image_from_sim(sim, var="p",output_image="image_af/pressure/test/"+sim+".png", airfoil=False, resolution=(1500, 750))
        get_image_from_sim(sim, var="Ux",output_image="image_af/velocity/uy/test/"+sim+".png", airfoil=False, resolution=(1500, 750))
        get_image_from_sim(sim, var="Uy",output_image="image_af/velocity/ux/test/"+sim+".png", airfoil=False, resolution=(1500, 750))

    # Load train dataset list
    train_dataset_list, train_dataset_names = af.dataset.load(root = PATH_TO_DATASET+"/Dataset", task = 'full', train = True)

    # Save initial conditions for train dataset
    # save_initial_conditions(train_dataset_names, output="image_af/train.json")

    for idx,sim in enumerate(train_dataset_names): 
        get_image_from_sim(sim, var="nut",output_image="image_af/viscosity/train/"+sim+".png", airfoil=False, resolution=(1500, 750))
        get_image_from_sim(sim, var="p",output_image="image_af/pressure/train/"+sim+".png", airfoil=False, resolution=(1500, 750))
        get_image_from_sim(sim, var="Ux",output_image="image_af/velocity/uy/train/"+sim+".png", airfoil=False, resolution=(1500, 750))
        get_image_from_sim(sim, var="Uy",output_image="image_af/velocity/ux/train/"+sim+".png", airfoil=False, resolution=(1500, 750))


def main():
    # ------ Original Dataset ----
    # Download Dataset
    # af.dataset.download(root = PATH_TO_DATASET, file_name = 'Dataset', unzip = True, OpenFOAM = False)
    
    # Create Dataset
    create_dataset()
    # ----------------------------












if __name__ == "__main__":
    # main()
    get_image_from_sim("airFoil2D_SST_31.382_3.588_1.994_6.206_0.0_13.271", var="nut", output_image="airfRANS_cropped_nut.png", airfoil=False, resolution=(1500, 750))
    get_image_from_sim("airFoil2D_SST_31.382_3.588_1.994_6.206_0.0_13.271", var="p", output_image="airfRANS_cropped_p.png", airfoil=False, resolution=(1500, 750))
    get_image_from_sim("airFoil2D_SST_31.382_3.588_1.994_6.206_0.0_13.271", var="Ux", output_image="airfRANS_cropped_ux.png", airfoil=False, resolution=(1500, 750)) 
    get_image_from_sim("airFoil2D_SST_31.382_3.588_1.994_6.206_0.0_13.271", var="Uy", output_image="airfRANS_cropped_uy.png", airfoil=False, resolution=(1500, 750))
        