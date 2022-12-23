class Environment{
    constructor(map, obstacles, cars){
        this.map = map || [];
        this.obstacles = obstacles || [];
        for(let i = 0; i < this.obstacles.length; i++){
            const o = obstacles[i];
            map.push([o[0], o[1], o[0] + o[2], o[1]]);
            map.push([o[0], o[1], o[0], o[1] + o[3]]);
            map.push([o[0] + o[2], o[1] + o[3], o[0] + o[2], o[1]]);
            map.push([o[0] + o[2], o[1] + o[3], o[0], o[1] + o[3]]);
        }
    }

    render(p){
        //Render Map
        for(let i = 0; i < mapWalls.length; i++)
            p.line(...mapWalls[i]);
    }
}