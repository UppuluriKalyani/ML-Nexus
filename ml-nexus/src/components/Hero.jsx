import React from 'react'
import Btn from './Btn'
import Card from './Card'
import { Swiper, SwiperSlide } from 'swiper/react'
import 'swiper/css';
import 'swiper/css/navigation';
import { Navigation } from 'swiper/modules';

function Hero() {
  const info = [
    {
      title: "Natural Language Processing (NLP)",
      img: "https://github.com/UppuluriKalyani/ML-Nexus/raw/main/img/nlp.jpg",
      desc: "Projects in this area involve working with text data, such as sentiment analysis",
      ref: "https://github.com/UppuluriKalyani/ML-Nexus/tree/main/Natural%20Language%20Processing"
    },
    {
      title: "Computer Vision",
      img: "https://github.com/UppuluriKalyani/ML-Nexus/raw/main/img/cv.jpg",
      desc: "Contributors can explore projects related to image classification object detection, facial recognition, and image segmentation using tools like OpenCV, convolutional neural networks (CNNs), and transfer learning.",
      ref: "https://github.com/UppuluriKalyani/ML-Nexus/tree/main/Computer%20Vision"
    },
    {
      title: "Neural Networks",
      img: "https://github.com/UppuluriKalyani/ML-Nexus/raw/main/img/nn.jpeg",
      desc: "Neural networks power most deep learning models. Contributions could include",
      ref: "https://github.com/UppuluriKalyani/ML-Nexus/tree/main/Neural%20Networks"
    },
    {
      title: "Generative Models",
      img: "https://github.com/UppuluriKalyani/ML-Nexus/raw/main/img/gm.jpeg",
      desc: "This includes working on projects related to Generative Adversarial Networks (GANs) for image generation, text-to-image models, or style transfer, contributing to fields like art creation and synthetic data generation.",
      ref: "https://github.com/UppuluriKalyani/ML-Nexus/tree/main/Generative%20Models"
    },
    {
      title: "Time Series Analysis",
      img: "https://github.com/UppuluriKalyani/ML-Nexus/raw/main/img/tsa.jpeg",
      desc: "Contributors can work on analyzing temporal data, building models for stock price prediction, climate forecasting, or IoT sensor data analysis using LSTM or GRU networks.",
      ref: "https://github.com/UppuluriKalyani/ML-Nexus/tree/main/Time%20Series%20Analysis"
    },
    {
        title: "Prediction Models",
        img: "https://github.com/UppuluriKalyani/ML-Nexus/raw/main/img/gm.jpeg",
        desc: "This includes working on projects related to Prediction Models.",
        ref: "https://github.com/UppuluriKalyani/ML-Nexus/tree/main/Prediction%20Models"
    },
  ];

  return (
    <div className="h-full w-full flex justify-center flex-col">
      <div className="flex mx-auto flex-col items-center">
        <h1 className='text-2xl mt-16 text-center flex items-center'>
          Welcome To <span className='dark:text-green-400 inline-block text-3xl lg:text-9xl'>ML Nexus</span>
        </h1>
        <p className='w-[90vw] lg:w-[65vw] text-center mt-4'>
          This repository is dedicated to providing top-quality machine learning tools and resources. Track our milestones, see top programming languages in use, and monitor community progress—all in one place.
        </p>
      </div>

      <div className="h-full w-full p-6 lg:p-10">
        <div className="flex justify-between mb-4 items-center">
          <h1 className='text-xl lg:text-2xl'>Hot Topics</h1>
          <Btn value={{name: "Contribute", ref: "https://github.com/UppuluriKalyani/ML-Nexus/issues"}} git={true} />
        </div>

        <div className="hidden lg:flex justify-evenly gap-6">
          {/* Desktop View with updated gap */}
          {info.map((data, i) => (
            <div
              key={i}
              className="card w-72 bg-white shadow-lg rounded-lg overflow-hidden transition-transform transform hover:translate-y-2 hover:shadow-xl hover:scale-105 ease-in-out duration-300"
            >
              <Card data={data} />
            </div>
          ))}
        </div>

        <div className="block lg:hidden">
          {/* Mobile View with Swiper */}
          <Swiper
            spaceBetween={10} // ensures spacing between slides
            navigation={{
              nextEl: '.swiper-button-next',
              prevEl: '.swiper-button-prev',
            }}
            slidesPerView={1}
          >
            {info.map((data, i) => (
              <SwiperSlide key={i}>
                <div
                  className="card w-full bg-white shadow-lg rounded-lg overflow-hidden transition-transform transform hover:translate-y-2 hover:shadow-xl hover:scale-105 ease-in-out duration-300"
                >
                  <Card data={data} />
                </div>
              </SwiperSlide>
            ))}
            <div className="swiper-button-next custom-arrow"></div>
            <div className="swiper-button-prev custom-arrow"></div>
          </Swiper>
        </div>
      </div>
    </div>
  );
}

export default Hero;
