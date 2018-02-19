const {assertClose} = require('./test');
const jsnet = require('../build/build');

function testMatmul() {
    const input1 = new jsnet.Variable(new jsnet.Tensor([2, 3], [0.5, 0.2, -0.2, -0.8, -0.3, 1]));
    const input2 = new jsnet.Variable(new jsnet.Tensor([3, 4], [0.3, -0.2, 0.4, -0.4, 0.1, -0.7,
        0.3, -0.2, 0.4, 0.6, -0.2, -0.1]));
    const result = jsnet.matmul(input1, input2);
    result.backward(new jsnet.Tensor([2, 4], [1, -0.7, -0.5, 0.5, 0.3, -0.3, 0.4, -1]));
    assertClose(input1.gradient, new jsnet.Tensor([2, 3], [0.04, 0.34, 0.03, 0.71,  0.56, -0.04]));
    assertClose(input2.gradient, new jsnet.Tensor([3, 4], [0.26, -0.11, -0.57, 1.05,
       0.11, -0.05, -0.22, 0.4, 0.1, -0.16, 0.5, -1.1]));
    assertClose(result.value, new jsnet.Tensor([2, 4],
        [0.09, -0.36, 0.3, -0.22, 0.13, 0.97, -0.61, 0.28]));
}

testMatmul();
console.log('PASS');
