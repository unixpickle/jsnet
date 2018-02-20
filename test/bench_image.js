var bench = require('./bench');
var jsnet = require('../build/build');

function benchmarkImagePatches() {
    const image = new jsnet.Variable(new jsnet.Tensor([2, 28, 28, 64]));
    const upstream = new jsnet.Tensor([2, 13, 13, 3, 3, 64]);
    bench('imagePatches', () => {
        const output = jsnet.imagePatches(image, 3, 3, 2, 2);
        output.backward(upstream);
    });
}

benchmarkImagePatches();
