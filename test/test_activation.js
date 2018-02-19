const {assertClose} = require('./test');
const jsnet = require('../build/build');

function testReLU() {
    const input = new jsnet.Variable(new jsnet.Tensor([3, 2], [1, 2, -3, 2, 0.1, -0.1]));
    const output = jsnet.relu(input);
    assertClose(output.value, new jsnet.Tensor([3, 2], [1, 2, 0, 2, 0.1, 0]));
    output.backward(new jsnet.Tensor([3, 2], [1, 2, 3, 4, 5, 6]));
    assertClose(input.gradient, new jsnet.Tensor([3, 2], [1, 2, 0, 4, 5, 0]));
}

testReLU();
console.log('PASS');
