const {assertClose} = require('./test');
const jsnet = require('../build/build');

function testReLU() {
    const input = new jsnet.Variable(new jsnet.Tensor([3, 2], [1, 2, -3, 2, 0.1, -0.1]));
    const output = jsnet.relu(input);
    assertClose(output.value, new jsnet.Tensor([3, 2], [1, 2, 0, 2, 0.1, 0]));
    output.backward(new jsnet.Tensor([3, 2], [1, 2, 3, 4, 5, 6]));
    assertClose(input.gradient, new jsnet.Tensor([3, 2], [1, 2, 0, 4, 5, 0]));
}

function testRsqrt() {
    const values = new jsnet.Tensor([2, 3], [0.55000, 0.15000, 1.96000, 0.87000, 0.53000, 0.94000]);
    const outputs = new jsnet.Tensor([2, 3], [1.34840, 2.58199, 0.71429, 1.07211, 1.37361, 1.03142]);
    const upstream = new jsnet.Tensor([2, 3], [-0.50000, 0.55000, 1.34000, -0.73000, -0.19000, 1.30000]);
    const grad = new jsnet.Tensor([2, 3], [0.61291, -4.73365, -0.24417, 0.44979, 0.24621, -0.71322]);

    const inputVar = new jsnet.Variable(values);
    const output = jsnet.rsqrt(inputVar);
    assertClose(output.value, outputs);
    output.backward(upstream);
    assertClose(inputVar.gradient, grad);
}

testReLU();
testRsqrt();
console.log('PASS');
