const {assertClose} = require('./test');
const jsnet = require('../build/build');

function testPool() {
    const input = new jsnet.Variable(new jsnet.Tensor([2], [2, 3]));
    const result = jsnet.pool(input, function(pooled) {
        return pooled;
    });
    result.backward(new jsnet.Tensor([2], [1, -1]));
    assertClose(input.gradient, new jsnet.Tensor([2], [1, -1]));
    assertClose(result.value, new jsnet.Tensor([2], [2, 3]));
}

function testScale() {
    const input = new jsnet.Variable(new jsnet.Tensor([2], [2, 3]));
    const result = jsnet.scale(input, 2);
    result.backward(new jsnet.Tensor([2], [1, -1]));
    assertClose(input.gradient, new jsnet.Tensor([2], [2, -2]));
    assertClose(result.value, new jsnet.Tensor([2], [4, 6]));
}

function testAddScalar() {
    const input = new jsnet.Variable(new jsnet.Tensor([2], [2, 3]));
    const result = jsnet.addScalar(input, 2);
    result.backward(new jsnet.Tensor([2], [1, -1]));
    assertClose(input.gradient, new jsnet.Tensor([2], [1, -1]));
    assertClose(result.value, new jsnet.Tensor([2], [4, 5]));
}

function testArithmetic() {
    const input1 = new jsnet.Variable(new jsnet.Tensor([2, 3], [0.5, -0.3, 0.2, 0.7, -0.8, 0.9]));
    const input2 = new jsnet.Variable(new jsnet.Tensor([2, 3], [0.1, -0.7, -0.1, 0.3, 0.5, -0.6]));
    const result = jsnet.add(jsnet.mul(input1, jsnet.div(input2, jsnet.sub(input1, input2))),
        input1);
    result.backward(new jsnet.Tensor([2, 3], [1, 2, 3, 4, 5, 6]));
    assertClose(input1.gradient, new jsnet.Tensor([2, 3],
        [0.9375, -4.125001, 2.6666665, 1.7499995, 4.260355, 5.04]));
    assertClose(input2.gradient, new jsnet.Tensor([2, 3],
        [1.5625, 1.1250004, 1.333333, 12.25, 1.8934911, 2.1599998]));
}

function testRepeated() {
    const input = new jsnet.Variable(new jsnet.Tensor([6, 1], [0.5, -0.3, 0.2, 0.7, -0.8, 0.9]));
    const result = jsnet.repeated(input, 2);
    result.backward(new jsnet.Tensor([2, 6, 1],
        [3, 2, 3, 4, 5, 6, -1, -2, -3, -4, 5, -7]));
    assertClose(result.value, new jsnet.Tensor([2, 6, 1], [0.5, -0.3, 0.2, 0.7, -0.8, 0.9,
        0.5, -0.3, 0.2, 0.7, -0.8, 0.9]));
    assertClose(input.gradient, new jsnet.Tensor([6, 1], [2, 0, 0, 0, 10, -1]));
}

function testSumOuter() {
    const input = new jsnet.Variable(new jsnet.Tensor([3, 2], [0.5, -0.3, 0.2, 0.7, -0.8, 0.9]));
    const result = jsnet.sumOuter(input);
    result.backward(new jsnet.Tensor([2], [1, 2]));
    assertClose(result.value, new jsnet.Tensor([2], [-0.1, 1.3]));
    assertClose(input.gradient, new jsnet.Tensor([3, 2], [1, 2, 1, 2, 1, 2]));
}

function testBroadcast() {
    const input = new jsnet.Variable(new jsnet.Tensor([2, 1], [3, 4]));
    let output = jsnet.broadcast(input, [3, 2, 1]);
    assertClose(output.value, new jsnet.Tensor([3, 2, 1], [3, 4, 3, 4, 3, 4]));
    output = jsnet.broadcast(input, [1, 2, 2, 1]);
    assertClose(output.value, new jsnet.Tensor([1, 2, 2, 1], [3, 4, 3, 4]));

    [[1], [2, 2], [3, 2, 2], [3, 1, 1]].forEach((newShape) => {
        if (jsnet.canBroadcast(input.value.shape, newShape)) {
            throw new Error('should not be able to broadcast to ' + newShape);
        }
    });
}

testPool();
testScale();
testAddScalar();
testArithmetic();
testRepeated();
testSumOuter();
testBroadcast();
console.log('PASS');
