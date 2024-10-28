import React from 'react'
import Nav from './Nav'
import Btn from './Btn'
import Card from './Card'
import Cards from './Cards'

function Home() {
  const info = [{
    title: "Natural Language Processing (NLP)",
    img: "https://github.com/UppuluriKalyani/ML-Nexus/raw/main/img/nlp.jpg",
    desc: "Projects in this area involve working with text data, such as sentiment analysis"
  },
  {
    title: "Computer Vision",
    img: "https://github.com/UppuluriKalyani/ML-Nexus/raw/main/img/cv.jpg",
    desc: "Contributors can explore projects related to image classification, object detection, facial recognition, and image segmentation using tools like OpenCV, convolutional neural networks (CNNs), and transfer learning."
  },
  {
    title: "Neural Networks",
    img: "https://github.com/UppuluriKalyani/ML-Nexus/raw/main/img/nn.jpeg",
    desc: "Neural networks power most deep learning models. Contributions could include"
  },
  {
    title: "Generative Models",
    img: "https://github.com/UppuluriKalyani/ML-Nexus/raw/main/img/gm.jpeg",
    desc: "This includes working on projects related to Generative Adversarial Networks (GANs) for image generation, text-to-image models, or style transfer, contributing to fields like art creation and synthetic data generation."
  },
  {
    title: "Time Series Analysis",
    img: "https://github.com/UppuluriKalyani/ML-Nexus/raw/main/img/tsa.jpeg",
    desc: "Contributors can work on analyzing temporal data, building models for stock price prediction, climate forecasting, or IoT sensor data analysis using LSTM or GRU networks."
  },
]
  return (
    <div className='h-screen overflow-hidden dark:bg-[#253341] w-full dark:text-[#AFC2CB]'>
      <Nav />
      <div className="h-full w-full flex justify-center flex-col">
      <div className=" flex mx-auto flex-col items-center">
      <h1 className='text-2xl mt-16 text-center flex items-center'> Welcome To <span className='dark:text-green-400 inline-block text-9xl'>ML Nexus</span></h1>
      <p className='w-[65vw] text-center mt-6'>This repository is dedicated to providing top-quality machine learning tools and resources. Track our milestones, see top programming languages in use, and monitor community progress—all in one place.</p>
      </div>
      <div className=" h-full w-full p-10">
       <div className="flex justify-between mb-2">
       <h1 className='text-2xl'>Hot Topics</h1>
       <Btn value={"Contribute"} git={true} />
       </div>
     <div className="flex justify-evenly">
    {info.map((info,i) => <Card data={info} key={i} />)}
     </div>
      </div>
      </div>
    </div>
  )
}

export default Home