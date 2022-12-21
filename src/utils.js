const del = () => localStorage.removeItem("car-model");

const euclideanDistance = (a, b) => {
    let dist = 0;
    for(let i = 0; i < a.length; i++)
        dist += Math.pow(b[i] - a[i], 2);
    return Math.sqrt(dist);
}

const closest =  (a, b) => {
        let closest = b[0];
        let minDist = euclideanDistance(a, b[0]);
        for(let i = 1; i < b.length; i++){
            const dist = euclideanDistance(a, b[i])
            if(dist < minDist){
                minDist = dist;
                closest = b[i];
            }
        }
        return closest;
    }

const intercept = (a, b, c, d) => {
    const p = a;
    const r = p.map((item, index) => b[index] - item);
    const q = c;
    const s = q.map((item, index) => d[index] - item);

    const RxS = r[0] * s[1] - r[1] * s[0];
    const u = ((q[0] - p[0]) * r[1] - (q[1] - p[1]) * r[0]) / RxS;
    const t = ((q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]) / RxS;

    if(RxS != 0 && u >= 0 && u <= 1 && t >= 0 && t <= 1)
        return [p[0] + t * r[0], p[1] + t * r[1]];
    return Infinity;
}