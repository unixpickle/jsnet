const {assertClose} = require('./test');
const jsnet = require('../build/build');

function testReLU() {
    const input = new jsnet.Variable(new jsnet.Tensor([3, 2], [1, 2, -3, 2, 0.1, -0.1]));
    const output = jsnet.relu(input);
    assertClose(output.value, new jsnet.Tensor([3, 2], [1, 2, 0, 2, 0.1, 0]));
    output.backward(new jsnet.Tensor([3, 2], [1, 2, 3, 4, 5, 6]));
    assertClose(input.gradient, new jsnet.Tensor([3, 2], [1, 2, 0, 4, 5, 0]));
}

function testPow() {
    const values = new jsnet.Tensor([2, 3], [0.14000, 2.29000, 0.87000, 0.34000, 0.32000, 0.65000]);
    const outputs = new jsnet.Tensor([2, 3], [10.58392, 0.37000, 1.18189, 3.64943, 3.92482, 1.67689]);
    const upstream = new jsnet.Tensor([2, 3], [0.90000, 0.06000, 0.72000, -0.15000, -0.04000, 0.36000]);
    const grad = new jsnet.Tensor([2, 3], [-81.64736, -0.01163, -1.17374, 1.93205, 0.58872, -1.11449]);

    const inputVar = new jsnet.Variable(values);
    const output = jsnet.pow(inputVar, -1.2);
    assertClose(output.value, outputs);
    output.backward(upstream);
    assertClose(inputVar.gradient, grad);
}

testReLU();
testPow();
console.log('PASS');
