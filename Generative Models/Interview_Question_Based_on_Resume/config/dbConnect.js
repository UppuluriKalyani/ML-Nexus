const mongoose = require('mongoose');

const dbConnect = async () => {
    try {
        await mongoose.connect(process.env.MONGO_URL,{
            useNewUrlParser: true,
            useUnifiedTopology: true,
        });
        console.log(`DB Connected Successfully`);
    } catch (error) {
        console.log(`Error${error}`);
    }
}

module.exports = dbConnect;