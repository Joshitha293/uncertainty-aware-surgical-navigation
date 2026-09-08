std_msgs/Header header
string structure_id
geometry_msgs/Point position

# Row-major 3x3 positional covariance in m^2.
float64[9] covariance

float64 physical_radius
float64 safety_margin

# Largest principal positional standard deviation in metres.
float64 maximum_principal_sigma

bool valid
