var bench = require('./bench');
var jsnet = require('../build/build');

function benchmarkMatmul() {
    const mat1 = new jsnet.Variable(new jsnet.Tensor([338, 576]));
    const mat2 = new jsnet.Variable(new jsnet.Tensor([576, 64]));
    const upstream = new jsnet.Tensor([338, 64]);
    bench('matmul', () => {
        const output = jsnet.matmul(mat1, mat2);
        output.backward(upstream);
    });
}

function benchmarkConv2d() {
    const image = new jsnet.Variable(new jsnet.Tensor([2, 28, 28, 64]));
    const filters = new jsnet.Variable(new jsnet.Tensor([3, 3, 64, 64]));
    const upstream = new jsnet.Tensor([2, 13, 13, 64]);
    bench('conv2d', () => {
        const output = jsnet.conv2d(image, filters, 2, 2);
        output.backward(upstream);
    });
}

benchmarkMatmul();
benchmarkConv2d();
