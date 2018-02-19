var DEFAULT_EPSILON = 1e-3;

function assertClose(t1, t2, epsilon) {
    epsilon = epsilon || DEFAULT_EPSILON;
    const diff = t1.copy().sub(t2);
    for (var i = 0; i < diff.data.length; ++i) {
        if (isNaN(diff.data[i])) {
            throw Error('NaN in Tensor');
        } else if (Math.abs(diff.data[i]) > epsilon) {
            throw Error('Tensors differ');
        }
    }
}

module.exports = {assertClose: assertClose};
