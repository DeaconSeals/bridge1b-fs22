import frame

# Wrapper function for basic bridge simulation. Takes as input a list of bridge
# points and named problem-instance parameters. Returns weight at which bridge
# failed or -100 if modifications were necessary. Base fitness function for
# Assignments 1a and 1b.
def basic_simulation(input_points, material:str, solid:bool, width:float, thickness=None, **kwargs):
    element_factory = frame.ElementFactory()

    assert material.casefold() in {'carbon fiber', 'wood', 'steel'}
    if material.casefold() == 'carbon fiber':
        element_factory.set_material_carbon_fiber()
    elif material.casefold() == 'wood':
        element_factory.set_material_wood()
    elif material.casefold() == 'steel':
        element_factory.set_material_steel()

    if solid:
        element_factory.set_cross_section_solid_square(width)
    else:
        assert thickness is not None
        element_factory.set_cross_section_hollow_square(width, thickness)

    results = frame.evaluate_frame(input_points=input_points, element_factory=element_factory, **kwargs)
    if results['invalid']:
        fitness = -100
    else:
        fitness = results['weight']

    return fitness, results['frame']

# Utility function for plotting bridges using matplotlib. Takes a truss object
# as input.
def plot_bridge(*args, **kwargs):
    frame.plot_frame(*args, **kwargs)