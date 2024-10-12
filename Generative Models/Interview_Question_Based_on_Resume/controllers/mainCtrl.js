const homepage = async (req,res) => {
    const locals = {
        title : 'PrepQuest',
        description: 'Interview Website',
    };

    res.render('index',{
        locals,
        layout: '../views/layouts/front-page',
    });
}

const about = async (req,res) => {
    const locals = {
        title : 'About - PrepQuest',
        description: 'Interview Website',
    };

    res.render('about',locals);
}

module.exports = { homepage, about }