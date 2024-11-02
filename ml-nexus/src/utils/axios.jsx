import axios from 'axios'

const instance = axios.create({
    baseURL: 'https://api.github.com/repos/UppuluriKalyani/ML-Nexus',
    headers: {
        'Authorization': import.meta.env.VITE_AUTH_KEY,
    }
})

export default instance