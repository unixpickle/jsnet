const {assertClose} = require('./test');
const jsnet = require('../build/build');

function testNormalizeChannels() {
    const values = new jsnet.Tensor([2, 3], [0.67000, 0.08000, 0.91000, 0.43000, 0.70000, 1.17000]);
    const outputs = new jsnet.Tensor([2, 3], [0.96699, -0.99484, -0.97167, -0.96699, 0.99484, 0.97167]);
    const upstream = new jsnet.Tensor([2, 3], [1.25000, 1.12000, 0.20000, 0.20000, 0.10000, -0.60000]);
    const grad = new jsnet.Tensor([2, 3], [0.27471, 0.01686, 0.16702, -0.27471, -0.01686, -0.16702]);

    testOp(values, outputs, upstream, grad, jsnet.normalizeChannels);
}

function testLogSoftmax() {
    const values = new jsnet.Tensor([4, 2, 3], [0.54000, 0.02000, 0.64000, 1.00000, 0.17000, 0.60000, 1.09000, 0.23000, 0.34000, 1.60000, 0.19000, 0.02000, 2.28000, 1.12000, 0.83000, 0.16000, 0.66000, 0.97000, 0.57000, 1.67000, 0.57000, 0.96000, 0.05000, 0.82000]);
    const outputs = new jsnet.Tensor([4, 2, 3], [-0.99314, -1.51314, -0.89314, -0.74497, -1.57497, -1.14497, -0.63950, -1.49950, -1.38950, -0.37165, -1.78165, -1.95165, -0.43700, -1.59700, -1.88700, -1.58855, -1.08855, -0.77855, -1.61027, -0.51027, -1.61027, -0.82061, -1.73061, -0.96061]);
    const upstream = new jsnet.Tensor([4, 2, 3], [-1.54000, -0.36000, 0.85000, -0.07000, -0.33000, 0.28000, 0.09000, -1.30000, -0.79000, -0.35000, -0.66000, -1.09000, 0.54000, 1.77000, -0.26000, 0.28000, -0.46000, 1.11000, -1.10000, 0.32000, -0.19000, -1.36000, -0.43000, -0.08000]);
    const grad = new jsnet.Tensor([4, 2, 3], [-1.15107, -0.12877, 1.27984, -0.01303, -0.30516, 0.31819, 1.14511, -0.85352, -0.29160, 1.09816, -0.30644, -0.79172, -0.78424, 1.35487, -0.57063, 0.09007, -0.77314, 0.68306, -0.90616, 0.90232, 0.00384, -0.53689, -0.09868, 0.63557]);

    testOp(values, outputs, upstream, grad, jsnet.logSoftmax);
}

function testOp(values, outputs, upstream, grad, fn) {
    const inputVar = new jsnet.Variable(values);
    const output = fn(inputVar);
    assertClose(output.value, outputs);
    output.backward(upstream);
    assertClose(inputVar.gradient, grad);
}

testNormalizeChannels();
testLogSoftmax();
console.log('PASS');
