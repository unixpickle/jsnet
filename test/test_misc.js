const {assertClose} = require('./test');
const jsnet = require('../build/build');

function testNormalizeChannels() {
    const values = new jsnet.Tensor([2, 3], [0.67000, 0.08000, 0.91000, 0.43000, 0.70000, 1.17000]);
    const outputs = new jsnet.Tensor([2, 3], [0.96699, -0.99484, -0.97167, -0.96699, 0.99484, 0.97167]);
    const upstream = new jsnet.Tensor([2, 3], [1.25000, 1.12000, 0.20000, 0.20000, 0.10000, -0.60000]);
    const grad = new jsnet.Tensor([2, 3], [0.27471, 0.01686, 0.16702, -0.27471, -0.01686, -0.16702]);

    const inputVar = new jsnet.Variable(values);
    const output = jsnet.normalizeChannels(inputVar);
    assertClose(output.value, outputs);
    output.backward(upstream);
    assertClose(inputVar.gradient, grad);
}

testNormalizeChannels();
console.log('PASS');
