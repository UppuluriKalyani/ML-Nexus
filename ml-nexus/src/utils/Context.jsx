import React, { useEffect, useState } from 'react'
import { createContext } from 'react'
import axios from '../utils/axios'

export const repoContext = createContext()

function Context(props) {

    const [projects, setProjects] = useState([])
    const [info, setInfo] = useState([])
    const [langs, setLangs] = useState([])

    const getRepo = async () => {
        try {
            const result = await axios.get('contents')
            if (!result) return console.log('not found')
            const projectsdata = result.data.filter((item, i) => item.type === 'dir' && item.name !== '.github')
            setProjects(projectsdata)

        } catch (error) {
            console.log(error)
        }
    }
    const getInfo = async () => {
        try {
            const result = await axios.get('')
            if (!result) return console.log('not found')
            const languagesResult = await axios.get('languages');
            if (!languagesResult) return console.log('Languages not found');
            const languages = languagesResult.data;
           
            const allLanguages = Object.entries(languages).map(([name,value]) => ({
                name,
                value
            }))
           
            setLangs(allLanguages)
            const mostUsedLanguage = Object.keys(languages).reduce((a, b) => languages[a] > languages[b] ? a : b);

            try {
                setInfo([
                    {
                        title: "Total Stars",
                        info: result.data.stargazers_count
                    },
                    {
                        title: "Total Forks",
                        info: result.data.forks_count
                    },
                    {
                        title: "Most Used Language",
                        info: mostUsedLanguage
                    },
                    {
                        title: "Open Issues",
                        info: result.data.open_issues
                    },
                    {
                        title: "Repository Size",
                        info: result.data.size
                    },
                    {
                        title: "License",
                        info: result.data.license.name
                    }
                ])
            } catch (error) {
                console.log(error)
            }
        } catch (error) {
            console.log(error)
        }
    }
    useEffect(() => {
        getRepo()
        getInfo()
    }, [])

    return (
        <repoContext.Provider value={{ projects, info, langs, }}>
            {props.children}
        </repoContext.Provider >
    )
}

export default Context