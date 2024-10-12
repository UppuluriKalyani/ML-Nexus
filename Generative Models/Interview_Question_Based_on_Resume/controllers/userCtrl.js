const User = require("../models/User");

const registerUser = async (req, res) => {
    try {
      const { username, email, password } = req.body;
  
      // Check if user with the same email already exists
      const existingUser = await User.findOne({ email });
      if (existingUser) {
        // User already exists
        const receivedSignal = 'User already exists. Please use a different email or Login';
        return res.render('login',{
          receivedSignal:receivedSignal,
          layout:'../views/layouts/blank'
        })
      }
      // If user does not exist, create a new user
      const newUser = new User({ username, email, password });
      await newUser.save();
  
      // User successfully registered
      const receivedSignal = 'Successfully Registered!\nPlease Login';
      return res.render('login',{
        receivedSignal:receivedSignal,
        layout:'../views/layouts/blank'
      })
  
    } catch (error) {
      // Handle any other errors
      console.error('Error registering user:', error);
      const receivedSignal = 'An error occurred while registering. Please try again later.';
      return res.render('login',{
        receivedSignal:receivedSignal,
        layout:'../views/layouts/blank'
      })
    }
};

const loginUser = async (req, res) => {
    try {
        res.redirect('/dashboard/');
    } 
    catch (error) {
        return res.send(error);
    }
};

const loginRender = async (req,res) => {
    try {
        return res.render('login',{
          receivedSignal:null,
          layout:'../views/layouts/blank'
        })
      } catch (error) {
        return res.send(error)
      }
}

module.exports = {registerUser,loginRender,loginUser}