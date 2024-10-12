const Skills = require("../models/Skills");

const dashboard = async (req,res) => {
    const locals = {
        title : 'Get Questions',
        description: 'Interview',
    };

    res.render('dashboard/index',{
        locals,
        layout: '../views/layouts/dashboard',
    });
}

const registerSkill = async (req, res) => {
    try {
        const {title, question, answer} = req.body;
        const newSkill = new Skills({title, question, answer});
        await newSkill.save();
        res.json(newSkill);
    } catch (error) {
        console.log(error);
    }
};

const searchSkill = async (req, res) => {
    const { skill } = req.body;
    
    try {
        const skillDoc = await Skills.findOne({ title: skill });
        if(!skillDoc){
            return res.status(404).json({message: "Skill not found"})
        }

        const locals = {
            title : 'Search Questions',
            description: 'Interview',
        };
    
        res.render('dashboard/search',{
            locals,
            layout: '../views/layouts/dashboard',
            results: {
                title: skillDoc.title,
                question: skillDoc.question,
                answers: skillDoc.answer
            }
        });
    } catch (error) {
        console.log(`Error Searching : ${error}`);
    }
};
module.exports = {dashboard, registerSkill, searchSkill}