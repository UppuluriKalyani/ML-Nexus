const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');

const userSchema = mongoose.Schema({
    username:{
        type: String,
        required: true,
    },
    email:{
        type: String,
        required: true,
    },
    password:{
        type: String,
        required: true,
    }
});

userSchema.pre('save', async function(next) {
    try {
      // Check if the password field is modified
      if (!this.isModified('password')) {
        return next();
      }
  
      // Hash the password with a salt rounds value of 10
      const hashedPassword = await bcrypt.hash(this.password, 10);
  
      // Replace the plain text password with the hashed password
      this.password = hashedPassword;
      next();
    } catch (error) {
      next(error); // Pass any error to the next middleware
    }
});

const User = mongoose.model('User', userSchema);
module.exports = User;