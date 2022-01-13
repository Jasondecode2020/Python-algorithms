function fn() {
    console.log(this);
}

fn.call("xx");