const mongoose = require('mongoose');

const skillsSchema = mongoose.Schema({
    title: {
        type: String,
        required: true,
    },
    question: {
        type: String,
        required: true,
    },
    answer: {
        type: String,
        required: true,
    }
});

const Skills = mongoose.model('Skills',skillsSchema);
module.exports = Skills;