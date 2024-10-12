const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const User = require('../models/User');

const authUser = async (req, res, next) => {
    try {
        const email = req.body.email;
        const password = req.body.password;

        const user = await User.findOne({ email });
        if(!user){
            return res.status(401).render('login',{
                receivedSignal: "Please check your Email and Password",
                layout: '../views/layouts/blank',
            })
        }

        const isMatch = await bcrypt.compare(password, user.password);
        if(!isMatch){
            return res.status(401).render('login',{
                receivedSignal: "Invalid Password",
                layout: '../views/layouts/blank',
            })
        }

        const token = jwt.sign(
            {_id: user._id, username:user.username}, 
            process.env.JWT_SECRET,
            { expiresIn: '30d' }
        );

        res.cookie('token',token,{
            httpOnly: true,
            maxAge: 30 * 24 * 60 * 60 * 1000,
            secure: process.env.NODE_ENV === 'production'
        });
        req.username = user.username;
        req.userId = user._id;
        next();

    } 
    catch (error) {
        console.error('Error authenticating user:', error);
        return res.status(500).render('login', {
            receivedSignal: "An error occurred. Please try again later.",
            layout: "../views/layouts/blank"
        });
    }
};

module.exports = {authUser};