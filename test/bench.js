var MIN_TIME = 1e3;

function benchmark(name, f) {
    var start = new Date().getTime();
    var count = 0;
    while (new Date().getTime() - start < MIN_TIME) {
        f();
        ++count;
    }
    console.log(name + ' took ' + ((new Date().getTime() - start) / count) + ' ms');
}

module.exports = benchmark;
