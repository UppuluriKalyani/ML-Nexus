import React, { useContext } from 'react';
import Stats from './Stats';
import { FaRegWindowMaximize } from 'react-icons/fa';
import { repoContext } from '../utils/Context';
import { SiJupyter } from "react-icons/si";
import { ImHtmlFive2 } from "react-icons/im";
import { TbBrandPython } from "react-icons/tb";
import { TbCircleLetterR } from "react-icons/tb";
import { TbBrandJavascript } from "react-icons/tb";
import { DiCss3 } from "react-icons/di";
import { IoLogoNodejs } from "react-icons/io";
import { VscTerminalPowershell } from "react-icons/vsc";
import { TbBrandTypescript } from "react-icons/tb";
import { TbBrandDocker } from "react-icons/tb";
import { SlPuzzle } from "react-icons/sl";
import Btn from './Btn';

function Statistics() {
  const { projects, info, langs } = useContext(repoContext);
  const totalBytes = langs.reduce((total, lang) => total + lang.value, 0);

  // Find the most used language based on bytes
  let mostUsedLanguage = { name: "", bytes: 0 };
  langs.forEach(lang => {
    if (lang.value > mostUsedLanguage.bytes) {
      mostUsedLanguage = { name: lang.name, bytes: lang.value };
    }
  });

  const colors = [
    "#A78BFA",
    "#92C9D1",
    "#34D399",
    "#F87171",
    "#FBBF24",
    "#F472B6",
    "#60A5FA",
    "#D4A5A5",
  ];

  const showIcon = (name) => {
    switch (name) {
      case "Jupyter Notebook":
        return <SiJupyter size="30px" />;
      case "HTML":
        return <ImHtmlFive2 size="30px" />;
      case "Python":
        return <TbBrandPython size="30px" />;
      case "JavaScript":
        return <TbBrandJavascript size="30px" />;
      case "R":
        return <TbCircleLetterR size="30px" />;
      case "CSS":
        return <DiCss3 size="30px" />;
      case "EJS":
        return <IoLogoNodejs size="30px" />;
      case "Shell":
        return <VscTerminalPowershell size="30px" />;
      case "TypeScript":
        return <TbBrandTypescript size="30px" />;
      case "Dockerfile":
        return <TbBrandDocker size="30px" />;
      case "Batchfile":
        return <FaRegWindowMaximize size="30px" />;
      default:
        return null;
    }
  };

  return (
    <>
      <div className="p-10">
        <h1 className="text-4xl font-bold">Repository Statistics</h1>
        <div className="md:flex-row flex items-center flex-col gap-8 p-10">
          {info && info.map((data, i) => <Stats data={data} key={i} index={i} />)}
        </div>

        <div className="md:px-20 px-2 md:flex md:flex-row flex-col flex gap-4">
          <div className="md:basis-1/2 md:h-[60vh] h-[50vh] dark:bg-[#324655] md:w-1/2 px-6 rounded-lg py-4 stats overflow-y-auto">
            <h1 className="text-2xl">Languages</h1>
            <div className="p-8">
              {langs.length > 0 &&
                langs.map((lang, i) => {
                  const percentage = ((lang.value / totalBytes) * 100).toFixed(2);
                  return (
                    <div key={i} className="flex items-center mt-7 gap-4">
                      {showIcon(lang.name)}
                      <h3 className="text-xl">{lang.name}</h3>

                      {/* Progress Bar */}
                      <div className="relative w-full mt-2">
                        <div
                          className="progress-bar-container"
                          style={{
                            backgroundColor: "#f0f0f0",
                            borderRadius: '5px',
                            height: '15px',
                            position: 'relative',
                          }}
                        >
                          <div
                            className="progress-bar"
                            style={{
                              width: `${percentage}%`,
                              backgroundColor: '#34D399',
                              borderRadius: '5px',
                              height: '100%',
                              position: 'relative',
                            }}
                          />
                        </div>

                        {/* Percentage Text outside to the right */}
                        <span
                          className="absolute font-bold text-sm text-center text-black right-0 top-1/2 transform -translate-y-1/2 mr-2"
                          style={{
                            whiteSpace: 'nowrap', // Prevents text from wrapping
                            textOverflow: 'ellipsis', // Adds ellipsis if text is too long
                          }}
                        >
                          {percentage}%
                        </span>
                      </div>
                    </div>
                  );
                })}
            </div>
          </div>

          {/* Milestones Progress */}
          <div className="basis-1/2 h-[60vh] dark:bg-[#324655] md:w-1/2 px-6 rounded-lg py-4 overflow-hidden relative">
            <h1 className="text-2xl">Milestones progress</h1>
            <div className="flex lg:flex-row flex-col lg:items-start p-4 gap-5 overflow-y-scroll max-h-[50vh] stats stats relative">
              <div className="sticky mx-auto top-0 left-0">
                <div className="relative bg-black lg:w-64 lg:h-64 w-32 h-32 rounded-full">
                  <div className="absolute inset-0 rounded-full bg-[conic-gradient(from_0deg,_#A78BFA_0%,_#A78BFA_10%,_#92C9D1_10%,_#92C9D1_20%,_#34D399_20%,_#34D399_25%,_#F87171_25%,_#F87171_40%,_#FBBF24_40%,_#FBBF24_50%,_#f472b6_50%,_#f472b6_70%,_#60A5FA_70%,_#60A5FA_90%,_#D4A5A5_90%,_#D4A5A5_100%)]"></div>
                </div>
              </div>
              <div className="flex flex-col text-md gap-6 h-full overflow-y-auto">
                {projects.map((p, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <div
                      className={`rounded-full h-3 w-3 lg:h-6 lg:w-6`}
                      style={{ backgroundColor: i > 6 ? colors[7] : colors[i] }}
                    ></div>
                    <h4 className="text-center">{p.name}</h4>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Raise Issue */}
        <div className="flex mt-4">
          <div className="mx-auto h-[30vh] dark:bg-[#3D5966] w-full px-6 flex items-center justify-center">
            <div className="flex flex-col items-center gap-4">
              <h1 className="text-2xl flex items-center gap-3">
                <span> <SlPuzzle /></span> Have Something in Mind?
              </h1>
              <Btn value={{ name: "Raise issue Now!", ref: "https://github.com/UppuluriKalyani/ML-Nexus/issues/new/choose" }} />
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default Statistics;
