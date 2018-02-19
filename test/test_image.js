const {assertClose} = require('./test');
const jsnet = require('../build/build');

function testPadImages() {
    const image = new jsnet.Tensor([2, 4, 2, 3], [0.510, -1.990, 0.680, -2.280, 0.910, 0.070, 0.800, -0.060, -0.540, 0.020, -1.990, 1.820, -0.050, 2.630, 1.630, 0.430, 1.050, 0.320, 0.790, -0.530, -0.350, -1.220, 0.060, -0.640, -1.300, 0.380, 0.380, 0.540, 0.130, -2.820, 0.340, -1.170, -0.360, 2.380, -1.760, -0.340, -0.240, 0.190, -0.170, -0.700, 0.910, -0.620, -0.970, -0.230, -0.330, 1.710, -1.470, -0.210]);
    const padded = new jsnet.Tensor([2, 7, 9, 3], [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.510, -1.990, 0.680, -2.280, 0.910, 0.070, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.800, -0.060, -0.540, 0.020, -1.990, 1.820, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, -0.050, 2.630, 1.630, 0.430, 1.050, 0.320, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.790, -0.530, -0.350, -1.220, 0.060, -0.640, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, -1.300, 0.380, 0.380, 0.540, 0.130, -2.820, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.340, -1.170, -0.360, 2.380, -1.760, -0.340, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, -0.240, 0.190, -0.170, -0.700, 0.910, -0.620, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, -0.970, -0.230, -0.330, 1.710, -1.470, -0.210, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]);
    const upstream = new jsnet.Tensor([2, 7, 9, 3], [0.500, 0.060, -0.640, -0.180, -1.590, -0.240, 0.060, -0.250, -0.770, 1.680, -2.370, 0.020, 0.940, -0.090, 1.090, 2.750, -2.090, 0.760, 1.950, -0.420, -0.730, -1.060, 0.550, -0.090, 0.690, -0.620, 0.450, 2.800, -0.540, -1.580, 0.930, 0.450, 0.060, 0.410, -0.930, -1.520, -0.890, -0.580, -0.080, -0.040, -0.430, -0.960, 0.640, 0.020, -0.250, 0.460, 0.000, 0.760, -1.110, -0.470, 1.560, 0.110, -0.560, -1.460, -0.810, -1.160, 1.190, 0.740, -0.550, -0.840, -2.050, 1.410, 0.370, -0.190, 0.380, -0.340, 0.370, 0.590, -0.400, 0.740, -0.560, -1.430, 0.150, 2.260, -0.250, 0.340, 0.240, -0.390, -0.170, 0.080, 0.110, 1.670, 0.570, -0.170, 0.300, 0.700, -0.930, 1.080, -0.500, -1.690, 0.360, 1.000, 0.160, -0.390, 1.150, -0.680, 0.500, 0.210, 0.090, -1.880, -0.760, -1.550, 1.150, -1.030, 0.030, 1.920, 0.040, -0.530, -1.610, 0.390, 1.090, -0.820, -0.990, 2.980, 0.620, 0.720, -1.130, 0.450, 0.710, 1.560, 0.020, -0.180, -2.330, -0.010, -0.750, -0.970, 0.720, -0.220, -0.780, -2.450, -2.180, -1.160, 0.110, 0.680, -0.190, -0.020, -0.330, -1.050, -1.440, 0.510, -2.360, -0.370, 1.740, -0.350, -0.270, -2.100, 0.400, -1.580, 0.180, -1.940, 1.530, -0.680, -2.430, -0.140, -0.250, -0.970, -0.910, 0.050, 0.150, -0.540, 1.650, 0.150, -0.500, 0.190, -1.810, -0.510, -0.010, 0.710, -0.450, 0.950, -0.870, -0.300, -2.130, -0.190, 1.310, 0.580, -1.110, -0.980, -0.400, -0.110, 0.200, 0.040, 0.720, 0.710, 1.320, 0.370, 1.820, 0.770, 0.330, 1.070, 0.110, 1.030, 1.190, -1.410, 0.780, -0.590, 1.240, 2.170, -1.380, -1.470, 0.130, -0.520, 1.180, -1.480, -0.240, 0.810, 1.640, 1.440, -0.720, -1.980, -0.240, -0.250, 0.250, 0.750, -0.270, 0.040, 0.860, 0.300, 1.290, -0.480, 0.750, 0.540, 0.310, -0.220, 0.420, -0.740, -0.240, -1.460, 0.450, -1.190, -1.210, 1.670, 0.050, 0.190, 0.010, -1.000, -0.140, 0.540, -0.710, 1.740, 1.040, 0.960, 0.130, 1.320, -0.510, -0.960, -1.300, -0.040, 0.310, -0.150, -0.990, -0.340, 0.660, 1.410, 0.070, -0.230, 1.230, -0.130, 1.390, 2.000, -1.090, 0.700, 0.870, -0.860, 0.760, 1.770, -0.040, 2.200, -0.480, 0.120, -0.480, -0.490, 0.120, -0.610, -0.580, -1.460, -0.390, 0.610, 1.920, -0.230, -0.350, 1.230, 1.750, 0.250, -0.660, -0.980, -1.100, -0.070, -1.490, -0.210, -0.080, 0.870, 1.520, -1.920, 1.120, 0.300, 1.240, 0.110, -0.720, -0.680, 0.030, 1.750, 0.750, 0.370, -0.350, 0.820, -0.390, -2.140, -2.120, 0.670, -0.480, 0.280, 0.130, -0.270, -0.810, -1.650, -0.700, -1.190, 1.770, 0.140, -0.210, 0.030, 1.270, -2.460, -0.440, -1.080, 0.270, 0.300, -0.950, 2.150, -1.120, 1.000, -1.110, 1.010, -2.360, -1.110, -0.380, -0.760, -1.000, 1.490, -0.040, 0.680, 1.750, 0.060, -2.560, -0.210, -0.370, 0.990, 1.150, -0.650, 1.330, -0.440, 1.030, 0.440, 0.310, 0.330, -0.930, -0.900, -0.020, 0.710, -0.170, -0.280, -0.600, 1.650, 0.080, -0.020, -0.680, -1.160, -0.340, -0.140, 1.560, 1.840, -1.660, -2.060, 0.980, -0.250, 0.040, 0.000]);
    const grad = new jsnet.Tensor([2, 4, 2, 3], [-0.890, -0.580, -0.080, -0.040, -0.430, -0.960, -0.190, 0.380, -0.340, 0.370, 0.590, -0.400, 0.360, 1.000, 0.160, -0.390, 1.150, -0.680, 0.450, 0.710, 1.560, 0.020, -0.180, -2.330, -0.740, -0.240, -1.460, 0.450, -1.190, -1.210, 0.660, 1.410, 0.070, -0.230, 1.230, -0.130, -0.230, -0.350, 1.230, 1.750, 0.250, -0.660, -0.390, -2.140, -2.120, 0.670, -0.480, 0.280])

    const imageVar = new jsnet.Variable(image);
    const actualOut = jsnet.padImages(imageVar, 1, 4, 2, 3)
    assertClose(actualOut.value, padded);
    actualOut.backward(upstream);
    assertClose(imageVar.gradient, grad);
}

function testImagePatches() {
    const image = new jsnet.Tensor([2, 5, 8, 3], [-0.380, 0.190, 0.500, -1.290, 0.710, 1.710, 0.170, 1.050, -0.050, -0.120, 0.230, -0.160, 1.130, 0.850, -1.610, -0.560, 0.920, -0.780, -0.440, -0.270, 0.710, 0.910, -0.700, -0.880, 1.450, 0.500, 1.630, 0.550, 0.630, -0.960, -0.100, -1.670, 0.480, -0.620, -1.280, -0.370, -0.180, -0.040, 0.580, 1.280, 0.190, -1.870, -0.920, 0.190, 0.220, 2.040, 1.430, -0.300, -0.970, -0.280, 1.170, -0.270, 0.000, 0.020, 1.570, 0.850, -0.820, 0.260, 0.090, -0.640, -0.080, 0.520, -2.410, -0.240, 1.130, -0.190, 0.620, -2.020, -1.380, -0.750, 0.570, -0.760, 0.190, 0.550, -0.120, -0.040, 0.620, 0.200, -0.040, -0.550, 0.470, 1.610, 0.260, 0.450, 0.510, 0.290, -0.550, -0.580, -0.600, -0.850, -1.330, 1.160, 1.620, -2.260, -0.230, 1.280, -0.530, 0.850, 0.640, -0.880, -0.390, -0.170, 0.540, -1.260, -0.460, 0.490, -0.540, -0.220, 0.830, -0.380, 0.670, -0.200, 0.150, 0.990, 0.020, -0.620, -0.830, -0.790, -1.480, 0.350, 1.390, -0.760, -0.960, -1.160, 0.810, -0.350, 0.280, 0.630, 1.200, -0.650, 0.650, -2.770, 1.980, -0.110, -1.050, -0.210, 2.170, 0.250, 0.980, 0.180, -0.430, -0.250, 1.110, 0.990, -1.170, 0.000, -0.510, -0.230, 0.930, 0.550, -0.720, 0.470, 1.390, -0.320, 0.650, 0.250, -1.290, -0.470, 0.370, 0.500, 0.320, -0.150, 0.120, 1.390, -0.820, 1.150, -0.150, -0.660, -0.970, -0.490, 0.320, 0.610, -1.180, 1.290, 1.180, 0.650, -0.110, 1.230, 0.000, 1.060, -0.430, 0.980, 1.490, 1.670, -0.340, 1.190, -1.670, 0.770, -0.070, -0.390, 0.070, 0.440, -0.920, -0.840, 0.160, -1.040, -0.580, -0.150, -0.060, 1.800, 0.440, -0.050, 1.320, -2.650, 1.350, 0.730, 0.080, -0.120, -0.020, 1.120, -0.370, -0.130, -0.470, 2.070, 0.700, 1.720, 1.260, -0.300, -0.200, -1.680, -0.400, -0.390, -0.080, 0.140, 0.040, 0.150, -0.680, 0.820, -1.430, 1.210, -0.200, -1.600, 0.910, -0.210, -0.350, 0.200, -0.250, 0.970, 1.180, 0.050]);
    const patches = new jsnet.Tensor([2, 2, 3, 18], [-0.380, 0.190, 0.500, -1.290, 0.710, 1.710, 1.450, 0.500, 1.630, 0.550, 0.630, -0.960, -0.970, -0.280, 1.170, -0.270, 0.000, 0.020, -0.120, 0.230, -0.160, 1.130, 0.850, -1.610, -0.620, -1.280, -0.370, -0.180, -0.040, 0.580, 0.260, 0.090, -0.640, -0.080, 0.520, -2.410, -0.440, -0.270, 0.710, 0.910, -0.700, -0.880, -0.920, 0.190, 0.220, 2.040, 1.430, -0.300, 0.620, -2.020, -1.380, -0.750, 0.570, -0.760, -0.970, -0.280, 1.170, -0.270, 0.000, 0.020, 0.190, 0.550, -0.120, -0.040, 0.620, 0.200, -0.530, 0.850, 0.640, -0.880, -0.390, -0.170, 0.260, 0.090, -0.640, -0.080, 0.520, -2.410, 1.610, 0.260, 0.450, 0.510, 0.290, -0.550, 0.490, -0.540, -0.220, 0.830, -0.380, 0.670, 0.620, -2.020, -1.380, -0.750, 0.570, -0.760, -1.330, 1.160, 1.620, -2.260, -0.230, 1.280, 0.020, -0.620, -0.830, -0.790, -1.480, 0.350, 1.390, -0.760, -0.960, -1.160, 0.810, -0.350, -1.170, 0.000, -0.510, -0.230, 0.930, 0.550, -0.970, -0.490, 0.320, 0.610, -1.180, 1.290, -0.650, 0.650, -2.770, 1.980, -0.110, -1.050, -0.320, 0.650, 0.250, -1.290, -0.470, 0.370, 1.230, 0.000, 1.060, -0.430, 0.980, 1.490, 0.980, 0.180, -0.430, -0.250, 1.110, 0.990, 0.120, 1.390, -0.820, 1.150, -0.150, -0.660, -1.670, 0.770, -0.070, -0.390, 0.070, 0.440, -0.970, -0.490, 0.320, 0.610, -1.180, 1.290, -0.920, -0.840, 0.160, -1.040, -0.580, -0.150, 1.260, -0.300, -0.200, -1.680, -0.400, -0.390, 1.230, 0.000, 1.060, -0.430, 0.980, 1.490, -0.050, 1.320, -2.650, 1.350, 0.730, 0.080, 0.150, -0.680, 0.820, -1.430, 1.210, -0.200, -1.670, 0.770, -0.070, -0.390, 0.070, 0.440, -0.370, -0.130, -0.470, 2.070, 0.700, 1.720, -0.350, 0.200, -0.250, 0.970, 1.180, 0.050]);
    const upstream = new jsnet.Tensor([2, 2, 3, 18], [0.760, -0.300, -0.370, -0.420, -0.450, 0.420, 1.150, 0.960, -0.340, -1.050, 1.430, -1.410, -0.120, -1.160, 0.240, -0.990, -0.450, 1.780, -0.350, -1.230, 0.430, 0.690, -0.420, -0.240, 0.160, -0.440, -0.300, 0.120, -0.410, 1.230, -1.440, 0.160, 1.020, 0.190, 0.710, 0.530, -1.590, -0.760, -0.700, 0.270, 1.450, -1.330, -0.200, 1.450, -0.230, -0.480, 1.230, 1.190, 1.590, 0.820, 1.170, -0.300, 0.540, -0.840, -0.140, 1.230, -0.770, 0.460, 0.300, 0.060, -1.320, 0.580, -0.740, -0.580, -0.550, -0.540, -0.330, -0.900, 0.370, -1.310, 0.500, -0.500, -0.150, -0.610, 0.810, -1.790, 0.720, -1.400, -0.040, -0.810, -0.190, 0.190, -0.330, 0.320, 0.230, 1.720, -2.770, 0.920, 0.140, -0.320, -0.640, 0.300, -2.060, 0.190, 3.030, 0.380, 0.770, 0.520, -0.730, -1.450, -0.130, 0.500, 2.280, -1.190, 0.600, 1.190, 0.200, 0.020, 0.530, -0.870, 0.130, 0.860, -0.290, 2.090, -0.360, 0.460, 0.890, -0.040, -0.610, -0.820, 0.410, -1.050, 0.060, 0.510, -0.790, -1.140, 0.280, 0.980, 0.290, 1.550, -0.340, -1.090, 0.680, 0.570, 1.390, 1.120, 2.460, -1.280, 0.280, 0.180, -1.430, 0.320, -0.990, -1.100, -0.570, 0.260, -3.250, 1.730, -0.390, -1.190, -0.510, -1.640, 2.090, -0.880, 0.750, 0.550, 0.780, -0.920, -0.070, -0.450, 1.780, 2.870, 0.090, -0.190, 1.000, 2.170, 0.030, 0.370, 1.190, -0.150, 0.000, -0.620, 0.740, 0.990, -0.260, -1.800, 0.410, -0.670, -0.370, -0.310, 0.070, -1.650, -0.480, -0.670, 0.580, 0.270, 0.750, 0.330, 0.990, 1.410, 0.990, 0.010, -0.210, 0.420, -0.630, 0.310, 3.320, -1.240, 0.290, -0.320, 0.420, -0.840, -0.160, -0.590, 0.830, -1.040, 0.240, -0.360, -0.300, 0.300, -1.850, 1.110, -0.040, 1.670, 0.620, -0.450]);
    const grad = new jsnet.Tensor([2, 5, 8, 3], [0.760, -0.300, -0.370, -0.420, -0.450, 0.420, 0.000, 0.000, 0.000, -0.350, -1.230, 0.430, 0.690, -0.420, -0.240, 0.000, 0.000, 0.000, -1.590, -0.760, -0.700, 0.270, 1.450, -1.330, 1.150, 0.960, -0.340, -1.050, 1.430, -1.410, 0.000, 0.000, 0.000, 0.160, -0.440, -0.300, 0.120, -0.410, 1.230, 0.000, 0.000, 0.000, -0.200, 1.450, -0.230, -0.480, 1.230, 1.190, -0.260, 0.070, -0.530, -0.530, -0.150, 1.840, 0.000, 0.000, 0.000, -1.590, -0.450, 1.830, -1.600, 1.430, -0.870, 0.000, 0.000, 0.000, 0.950, 1.120, -0.890, -0.110, 3.570, -0.460, -1.320, 0.580, -0.740, -0.580, -0.550, -0.540, 0.000, 0.000, 0.000, -0.040, -0.810, -0.190, 0.190, -0.330, 0.320, 0.000, 0.000, 0.000, 0.770, 0.520, -0.730, -1.450, -0.130, 0.500, -0.330, -0.900, 0.370, -1.310, 0.500, -0.500, 0.000, 0.000, 0.000, 0.230, 1.720, -2.770, 0.920, 0.140, -0.320, 0.000, 0.000, 0.000, 2.280, -1.190, 0.600, 1.190, 0.200, 0.020, 0.530, -0.870, 0.130, 0.860, -0.290, 2.090, 0.000, 0.000, 0.000, 0.280, 0.980, 0.290, 1.550, -0.340, -1.090, 0.000, 0.000, 0.000, -0.570, 0.260, -3.250, 1.730, -0.390, -1.190, -0.360, 0.460, 0.890, -0.040, -0.610, -0.820, 0.000, 0.000, 0.000, 0.680, 0.570, 1.390, 1.120, 2.460, -1.280, 0.000, 0.000, 0.000, -0.510, -1.640, 2.090, -0.880, 0.750, 0.550, 0.500, -1.240, 1.060, 2.680, -0.760, -0.770, 0.000, 0.000, 0.000, 0.350, -1.470, -1.910, -0.350, -0.410, -0.830, 0.000, 0.000, 0.000, 1.070, -1.240, 0.350, -1.290, 1.620, 2.280, 1.190, -0.150, 0.000, -0.620, 0.740, 0.990, 0.000, 0.000, 0.000, 0.750, 0.330, 0.990, 1.410, 0.990, 0.010, 0.000, 0.000, 0.000, 0.830, -1.040, 0.240, -0.360, -0.300, 0.300, -0.260, -1.800, 0.410, -0.670, -0.370, -0.310, 0.000, 0.000, 0.000, -0.210, 0.420, -0.630, 0.310, 3.320, -1.240, 0.000, 0.000, 0.000, -1.850, 1.110, -0.040, 1.670, 0.620, -0.450]);

    const imageVar = new jsnet.Variable(image);
    const actualOut = jsnet.imagePatches(imageVar, 3, 2, 2, 3);
    assertClose(actualOut.value, patches.reshape([2, 2, 3, 3, 2, 3]));
    actualOut.backward(upstream.reshape([2, 2, 3, 3, 2, 3]));
    assertClose(imageVar.gradient, grad);
}

function testConv2d() {
    const image = new jsnet.Tensor([2, 5, 8, 4], [0.640, -0.550, 0.630, -1.130, 0.610, 0.060, -0.760, 0.180, -0.310, 2.640, -0.790, 1.520, -1.730, -1.020, -0.960, 0.480, -0.360, 0.790, -0.020, -2.260, 1.270, -0.390, 0.220, 1.030, 0.060, 0.070, 1.220, -1.020, 0.900, -0.050, -0.120, -1.070, -1.060, 0.640, -0.900, 0.720, 0.070, 0.670, 0.720, -1.190, 1.580, -0.200, 0.870, 0.260, -0.280, -0.560, 2.000, -0.180, 0.690, 0.750, 1.610, 1.040, -0.760, -0.430, 0.090, 1.000, -0.870, -2.650, 0.150, 0.660, 1.080, -0.350, -1.090, -0.940, -0.100, 1.150, -0.830, -1.950, 0.090, -0.220, -0.530, -0.730, -0.810, 0.340, -1.140, 1.160, 0.500, 0.500, -1.220, 0.920, -0.480, 0.040, 0.320, 0.300, 0.710, 0.410, -0.470, -0.730, 0.380, -0.280, 1.760, 0.700, -1.960, 1.140, -1.350, 0.740, 0.460, 0.120, 0.150, -0.460, -1.030, -0.970, -0.600, -1.520, -1.120, 0.460, -0.690, 0.390, 0.500, 0.550, -0.410, 0.670, -2.040, -2.680, 0.330, 0.640, 1.690, -1.600, -0.630, 0.270, -0.380, -0.520, -0.740, 1.350, -0.140, -1.190, 1.590, 0.690, -0.290, 0.420, 0.840, 1.380, -0.530, -1.930, 1.490, -0.650, -1.340, -0.570, -1.260, -0.870, 0.740, 1.180, 0.270, 0.020, -0.240, 0.500, -0.430, -0.300, -0.720, 0.250, 0.070, -0.180, -0.520, -2.020, 0.650, 0.350, 0.170, -0.920, 0.160, -0.100, 0.720, 0.530, 1.560, 0.290, 1.110, -0.420, -1.920, -0.310, 0.860, -1.180, 1.350, 0.860, -0.750, -0.330, 1.250, -0.530, -0.730, 1.620, -0.350, -0.670, 0.420, 0.400, -1.520, 0.980, 1.480, 1.200, 0.420, -0.880, -0.100, -1.210, 1.720, 1.040, -0.250, -1.700, -0.490, -0.960, 2.120, -0.830, -0.060, -0.680, 0.090, 0.060, 0.120, -0.190, 1.250, -0.980, 0.710, 0.140, 0.750, 1.720, -0.540, 1.450, 0.560, 0.970, -1.680, 0.380, 0.230, 0.330, 1.350, -0.460, 0.620, -0.070, -0.550, 1.180, 0.500, 0.710, -0.550, 0.920, -0.810, 1.340, -0.250, -0.670, -0.320, 0.380, -0.620, 0.000, -1.170, -0.180, 0.710, -1.180, -0.270, 0.160, -0.390, 1.460, -0.900, 0.370, 1.370, -0.090, 0.820, 0.340, 0.120, 0.370, 1.920, -1.810, -0.050, -0.290, 0.520, -1.740, -1.090, 1.000, 0.040, -0.110, 1.550, -1.230, -0.430, 1.200, 0.040, -0.560, -1.790, -1.960, -1.720, -0.270, 0.030, 1.200, -0.710, -0.010, 0.940, 0.800, -1.650, -0.720, -0.630, -0.820, 1.000, 1.140, 2.130, -0.550, 0.610, -0.090, -0.960, -0.400, 1.100, -2.410, -1.840, 0.140, -0.040, -0.230, 1.310, -1.360, 0.220, 1.010, -0.240, 0.890, 0.550, -0.910, 0.170, -0.970, -1.150, 0.270, -0.980, -1.280, -0.600, 1.930, 0.470, -1.390, -0.550, 0.860, 0.690, 2.900, -0.350, 1.690]);
    const filters = new jsnet.Tensor([2, 3, 4, 5], [-1.130, -0.950, 0.990, 0.550, 2.040, 1.080, -0.740, 0.470, -0.330, 0.040, 1.220, 0.260, 0.850, 0.900, 1.870, -4.350, -0.790, 0.470, -0.800, 0.280, 0.380, -1.420, 0.810, 1.300, 0.370, -0.640, 1.090, -0.130, -0.900, -0.860, 0.540, -0.150, -0.510, -0.890, 1.790, 1.040, -1.180, -0.130, 0.730, -0.610, -0.610, -0.530, -1.080, -0.110, 1.000, -0.240, -0.200, -0.640, -0.690, -0.820, -1.080, 0.660, -0.170, -0.140, -0.560, 1.620, 2.660, 0.960, -2.940, -0.790, 0.990, -1.300, 0.560, 1.740, -2.440, -0.340, 2.270, -2.320, -4.290, -2.370, -0.520, 0.240, 1.360, 0.650, -2.290, -0.930, -1.770, 0.540, -0.060, 0.650, -1.540, -0.270, 3.560, -0.640, -0.690, -0.040, -0.140, -1.450, -1.210, 0.150, 2.830, -0.660, 0.400, 0.280, 1.820, -1.680, 0.010, -1.020, -0.400, -2.280, 0.080, 0.290, 0.690, 0.470, 1.260, -1.590, -0.080, 0.320, 0.350, -1.970, 0.480, -1.490, 0.080, 1.580, -0.190, 0.060, -0.270, 1.140, 1.230, 0.240]);
    const outputs = new jsnet.Tensor([2, 2, 6, 5], [10.519, 2.963, 0.727, -5.623, 7.572, 2.118, 2.214, 7.228, -2.093, -9.569, -0.956, -14.602, 3.354, 13.102, -4.273, -3.176, 6.059, 2.159, -4.827, -6.426, 12.607, -4.768, -3.880, 5.974, -4.551, -9.391, -3.059, 0.189, 3.053, 12.217, 13.594, 5.717, -0.001, -1.056, -5.107, 3.781, 5.028, -4.886, 0.283, -0.635, -4.615, 0.161, -3.293, -7.807, -3.625, -1.047, -2.763, -6.298, 1.561, 4.871, -8.394, -0.108, 15.684, 6.882, 7.121, 9.046, -8.852, 3.948, 9.932, -0.160, -1.397, -1.803, 16.734, 9.218, 7.323, 3.901, -9.063, 3.378, 11.338, 1.527, -7.717, -0.869, 8.545, 2.595, 2.917, -1.399, 5.133, 3.361, -2.374, -8.672, -4.996, -5.032, -7.067, -3.799, -13.437, -1.937, 2.594, 1.343, -8.115, 3.349, -0.513, -6.564, 5.400, 4.371, 7.037, 6.565, 3.627, -3.836, -1.056, -4.788, -2.405, 13.497, -7.020, -11.423, -7.438, 2.986, 3.420, -1.904, 1.588, 6.293, -9.531, 2.359, -2.865, -10.839, -1.110, 8.307, -0.814, -4.808, 3.564, 6.660]);
    const upstream = new jsnet.Tensor([2, 2, 6, 5], [-0.560, -0.390, 0.890, -0.780, -0.440, -0.320, -2.140, -0.990, -0.010, -0.920, 1.440, -1.480, -0.220, -1.050, -0.750, 2.050, -0.220, -2.330, -0.440, -0.550, -1.040, -0.460, -1.490, -0.070, -1.260, 0.970, 0.670, 0.960, -0.390, -0.710, -0.170, -0.970, -0.040, 1.020, 1.490, -0.990, -0.380, 0.230, 0.180, -0.440, -1.430, -0.590, -0.210, -0.360, -0.590, 1.300, 0.600, 1.030, -0.100, 0.250, -1.230, -0.580, -0.510, 1.050, -0.420, -0.530, -0.050, 1.550, -0.490, 2.350, 0.320, -0.560, 0.690, 0.250, 0.780, 0.200, 0.690, 0.120, -0.560, -0.650, -0.590, 2.620, -0.460, -0.600, 0.960, -0.440, -0.700, 1.020, -0.460, 1.280, -0.330, 0.140, 0.470, -0.760, 0.580, 0.650, 1.320, -0.700, 0.690, -0.380, -1.390, 1.240, -0.380, -1.420, 1.560, -0.220, -1.080, -1.310, -0.820, 0.540, 0.560, -1.800, 1.120, 0.140, -0.100, 1.250, -1.670, 0.630, 0.370, 0.110, 1.420, -0.770, 0.280, 0.430, 0.720, 1.110, 0.360, -0.100, -0.880, -0.350]);
    const imageGrad = new jsnet.Tensor([2, 5, 8, 4], [0.558, 0.342, -1.553, 3.663, -0.583, 1.637, -4.309, 1.829, -1.552, 2.206, -1.596, -0.143, -3.470, 2.388, -0.639, -13.082, -4.534, -0.460, -5.266, 8.463, -3.070, 4.406, -3.024, -1.585, 0.660, 3.323, 0.798, -2.896, -2.651, -0.130, -0.316, 5.983, 0.167, 1.629, 1.909, 1.452, 9.077, -0.565, -1.584, 4.298, 0.952, 6.365, -2.700, 4.386, -2.318, 11.199, 3.106, -3.989, -10.949, 8.698, 6.497, -0.714, -3.391, 1.424, -2.679, 1.008, -0.383, 2.512, 1.481, -2.853, -0.144, -0.026, -0.937, 0.322, 4.675, 0.238, 3.211, 1.088, 4.066, -3.900, 0.062, 5.256, 3.027, -2.371, -6.473, -0.571, -1.029, 3.021, 2.683, -8.737, 2.457, -0.611, 0.269, 2.333, 5.896, -1.141, 1.299, 9.405, 2.431, -1.139, 4.694, -9.265, 1.080, -2.444, -0.972, 0.081, -0.791, -9.958, -2.948, 2.761, -0.270, -1.599, 5.001, -2.068, 4.716, -0.191, 0.913, 6.073, 2.051, 1.766, -4.207, 2.975, 2.154, -1.405, 5.292, -2.950, -5.069, -9.773, -7.494, 7.121, 4.386, 1.760, 5.265, -5.161, 3.743, -3.458, -1.277, 1.710, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.582, 1.033, 2.515, -0.607, -0.307, -1.880, -0.116, -0.482, -2.315, -2.103, -1.200, -1.265, -1.417, 4.389, 4.565, 3.438, 3.493, -1.727, 4.473, 8.013, -2.926, -1.498, 0.915, -7.590, -1.155, 0.619, -0.634, 2.010, -0.796, -0.136, 0.404, 2.164, -0.037, -5.902, -0.986, 1.558, 1.396, 4.042, 4.390, -4.857, -5.455, 6.123, -2.170, -0.971, -5.376, -2.923, -5.470, 1.714, 2.249, -0.909, -4.506, -4.082, 4.755, 1.058, 0.293, -3.483, -3.317, -0.832, -1.451, -0.092, -0.203, -0.373, -0.548, -0.358, 2.418, -2.066, -0.057, 6.461, -3.237, 2.465, 1.922, -2.846, 3.899, 1.850, 5.223, 3.284, 7.798, 1.488, 1.423, -3.866, 4.275, -1.514, 2.029, -5.326, -0.697, -2.344, -1.111, -5.317, -1.532, -0.465, -1.800, -0.998, -1.013, 0.620, -0.625, 5.523, -9.478, 6.564, -3.992, -0.008, -2.006, 5.286, -6.159, 1.555, 1.413, -5.687, -3.646, 1.765, 6.787, -10.438, 2.873, -1.840, 2.246, -7.734, 6.564, -0.578, -1.493, 2.770, 9.375, -4.665, -0.165, -2.312, 4.374, 0.705, -0.730, -1.444, -1.335, -1.311, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]);
    const filterGrad = new jsnet.Tensor([2, 3, 4, 5], [-3.742, 8.886, 6.498, -0.200, 1.667, 0.716, -8.830, -0.045, -3.101, 2.409, -2.193, 2.423, 4.410, -0.074, 3.659, 7.688, 7.455, -0.523, -2.456, -1.996, -4.411, 5.144, -1.785, 3.087, -3.411, -2.757, -4.314, -4.465, -0.917, 1.175, 5.554, 10.762, 3.244, 0.354, 2.935, -5.849, -6.520, 3.488, -1.669, -3.508, 7.986, 5.620, -0.065, -0.160, -4.540, -5.493, 1.627, 7.367, -4.670, 6.424, 3.522, 1.198, -7.514, 3.831, -8.448, -3.596, -4.914, 3.561, 6.307, 4.768, 3.358, 5.653, 1.020, -7.165, 6.501, 4.941, 1.230, -2.146, -0.605, -7.122, -0.269, -6.029, -11.687, -0.700, -4.923, -2.371, 6.876, 2.365, -2.227, 0.502, -4.244, 0.308, -5.970, 1.178, 1.374, -4.041, -4.473, -6.253, -2.007, 1.807, 4.527, 7.981, -11.152, -6.742, -6.089, 1.266, 0.613, 0.067, -2.403, -2.628, 9.514, 7.858, 10.798, -3.644, -2.784, 2.194, 5.186, 6.380, 0.900, 7.090, -0.143, -2.182, -0.638, -4.543, -2.563, -0.144, 1.062, -4.378, 1.239, 0.266]);

    const imageVar = new jsnet.Variable(image);
    const filterVar = new jsnet.Variable(filters);
    const actualOut = jsnet.conv2d(imageVar, filterVar, 2, 1);
    assertClose(actualOut.value, outputs);
    actualOut.backward(upstream);
    assertClose(imageVar.gradient, imageGrad);
    assertClose(filterVar.gradient, filterGrad);
}

testPadImages();
testImagePatches();
testConv2d();
console.log('PASS');
