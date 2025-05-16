"""
Sample dataset for text summarization using ACO.
Contains short documents and their human-written summaries.
"""

DOCUMENTS = [
    {
        "id": 1,
        "title": "Artificial Intelligence",
        "text": """Artificial Intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. 
        AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
        The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving". 
        This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.
        AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars, automated decision-making, and competing at the highest level in strategic game systems.
        As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect.
        """,
        "summary": """Artificial Intelligence (AI) is machine-demonstrated intelligence, defined as the study of intelligent agents that perceive their environment and act to achieve goals. 
        The field has moved away from defining AI as mimicking human cognitive skills toward concepts of rationality. 
        Applications include search engines, recommendation systems, speech recognition, self-driving cars, and strategic games. 
        As AI capabilities advance, previously "intelligent" tasks are often no longer considered AI."""
    },
    {
        "id": 2,
        "title": "Climate Change",
        "text": """Climate change refers to significant changes in global temperature, precipitation, wind patterns, and other measures of climate that occur over several decades or longer.
        The Earth's climate has changed throughout history. Just in the last 650,000 years, there have been seven cycles of glacial advance and retreat, with the abrupt end of the last ice age about 11,700 years ago marking the beginning of the modern climate era — and of human civilization.
        Most of these climate changes are attributed to very small variations in Earth's orbit that change the amount of solar energy our planet receives.
        The current warming trend is of particular significance because it is unequivocally the result of human activity since the mid-20th century and proceeding at a rate that is unprecedented over millennia.
        It is undeniable that human activities have warmed the atmosphere, ocean, and land and that widespread and rapid changes in the atmosphere, ocean, cryosphere, and biosphere have occurred.
        The effects of climate change include rising sea levels, regional changes in precipitation, more frequent extreme weather events such as heat waves, and expansion of deserts.
        Future climate change effects are expected to include loss of biodiversity, stresses to existing food-producing systems, increased risks of drought and flooding, and harm to human health.
        """,
        "summary": """Climate change involves significant long-term alterations in temperature, precipitation, and other climate measures. 
        While Earth's climate has naturally changed throughout history, the current warming is significant because it results from human activity since the mid-20th century and is occurring at an unprecedented rate. 
        Effects include rising sea levels, changing precipitation patterns, extreme weather events, and desert expansion. 
        Future impacts may include biodiversity loss, food system stress, increased drought and flooding risks, and health consequences."""
    },
    {
        "id": 3,
        "title": "Quantum Computing",
        "text": """Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement, to perform calculations.
        The devices that perform quantum computations are known as quantum computers. Though current quantum computers are too small to outperform usual (classical) computers for practical applications, they are believed to be capable of solving certain computational problems, such as integer factorization, substantially faster than classical computers.
        The study of quantum computing is a subfield of quantum information science. It is not the same as quantum physics or quantum mechanics, although aspects of quantum physics are used to develop quantum computing technology.
        Quantum computing began in the early 1980s, when physicist Paul Benioff proposed a quantum mechanical model of the Turing machine.
        Richard Feynman and Yuri Manin later suggested that a quantum computer had the potential to simulate things that a classical computer could not.
        In 1994, Peter Shor developed a quantum algorithm for factoring integers that had the potential to decrypt RSA-encrypted communications.
        Despite ongoing experimental progress since the late 1990s, most researchers believe that "fault-tolerant quantum computing [is] still a rather distant dream."
        In recent years, investment in quantum computing research has increased in both the public and private sectors.
        """,
        "summary": """Quantum computing uses quantum states' properties like superposition and entanglement to perform calculations. 
        While current quantum computers cannot yet outperform classical computers for practical applications, they show promise for solving certain problems much faster. 
        This field emerged in the 1980s with theoretical proposals, followed by Shor's 1994 factoring algorithm with encryption implications. 
        Despite experimental progress, fault-tolerant quantum computing remains a challenge, though research investment has increased in both public and private sectors."""
    }
]
