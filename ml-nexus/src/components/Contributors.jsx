import React, { useContext } from 'react'
import { repoContext } from '../utils/Context'

function Contributors() {
    const { contributors } = useContext(repoContext)
    return (
        <>
            <div className="p-10 ">
                <h1 className='text-4xl font-bold text-center mb-6'>Our Contributors</h1>
                <div className="px-4 mt-6 grid grid-cols-9 sm:grid-cols-6 md:grid-cols-4 lg:grid-cols-5 gap-3 justify-items-center">
                    {contributors.map((c, i) => (
                        <div key={i} className="relative md:h-48 w-48 bg-[#324655] rounded-lg flex flex-col items-center justify-between shadow-lg transition-transform transform hover:scale-105 border-2 border-transparent hover:border-blue-500">
                            <div className="h-2/3 w-2/3 rounded-full overflow-hidden bg-gray-700 mt-2">
                                <img className='h-full w-full object-cover' src={c.avatar_url} alt={c.login} />
                            </div>
                            <a href={c.html_url} target="_blank" rel="noopener noreferrer" className='md:text-base text-xs mb-2 bg-teal-800 px-4 py-2 rounded-full text-center'>
                                {c.login}
                            </a>
                        </div>
                    ))}
                </div>
            </div>
        </>
    )
}

export default Contributors