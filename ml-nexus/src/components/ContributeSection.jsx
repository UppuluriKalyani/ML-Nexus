import React from 'react';
import Btn from './Btn';

const ContributeSection = () => {
  return (
    <section className="py-20 px-6 bg-[#EDF7F6] dark:bg-[#253341] text-[#28333F] dark:text-[#AFC2CB] animate-fadeIn">
      <div className="max-w-4xl mx-auto bg-white dark:bg-[#324655] rounded-xl p-12 text-center shadow-lg transition-shadow duration-300 hover:shadow-2xl border-2 border-transparent border-animated">
        <h2 className="text-3xl md:text-4xl font-bold mb-6 animate-slideIn">
          Each and Every Contribution Matters
        </h2>
        <p className="text-lg md:text-xl mb-10 animate-fadeInDelay">
          Join us in building something great. Your unique skills and knowledge can make a difference!
        </p>
        <Btn className='mx-auto md:w-1/3 w-40 flex items-center justify-center bg-blue-500 text-white rounded-lg shadow hover:bg-blue-600 hover:scale-110 transition-all duration-300 animate-bounceOnHover' value={{name:"Contribute Now", ref: "https://github.com/UppuluriKalyani/ML-Nexus/issues"}} />
      </div>
    </section>
  );
};

export default ContributeSection;
