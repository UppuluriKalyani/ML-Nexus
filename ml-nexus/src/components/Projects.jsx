import React, { useContext } from 'react';
import Btn from './Btn';
import { repoContext } from '../utils/Context';
import { motion } from 'framer-motion';

function Projects() {
  const { projects } = useContext(repoContext);

  return (
    <motion.div
      className="p-2 max-w-7xl mx-auto"
    >
      <h1 className="text-4xl font-extrabold leading-tight text-white mb-8 text-center">
        Projects
      </h1>

      {/* Grid Container: Controls the number of columns in each row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-7">
        {projects.map((p, i) => (
          <motion.div
            whileHover={{
              scale: 1.05,
              backgroundColor: '#3A5064',
            }}
            key={i}
            className="py-6 px-6 bg-[#324655] rounded-lg transition-all duration-300 ease-in-out transform hover:scale-105 hover:bg-[#3A5064] hover:shadow-xl flex flex-col justify-between items-center"
          >
            <h2 className="text-xl font-semibold text-white text-center mb-4">
              {p.name}
            </h2>
            <Btn
              className="rounded-md bg-blue-500 text-white py-2 px-4 mt-4 transition-colors duration-300 ease-in-out hover:bg-blue-600"
              value={{ name: 'View Models', ref: p.html_url }}
            />
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

export default Projects;
