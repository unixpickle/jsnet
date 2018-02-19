if [ -d build ]; then
    rm -r build
fi

mkdir build

echo '(function() {' >build/build.js
for file in tensor.js primitives.js image.js linear_ops.js export.js; do
    cat src/$file >>build/build.js
done
echo '})();' >>build/build.js
