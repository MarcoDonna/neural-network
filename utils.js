function dot(a, b){
    if(a.length != b.length)
        throw new Error('Invalid shape');

    let ret = 0;
    for(let i = 0; i < a.length; i++)
        ret += a[i] * b[i];
    return ret;
}

function add(vector, value=0){
    return vector.map(item => item+value);
}

function sum(vector){
    return vector.reduce((acc, val) => acc + val);
}

function avg(vector){
    return sum(vector) / vector.length;
}