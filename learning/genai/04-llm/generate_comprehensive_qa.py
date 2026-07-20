"""
Generate comprehensive Q&A training data from every chapter of The Everglades Cipher.
Produces large RLHF and instruction datasets covering the full novel.
Run: python generate_comprehensive_qa.py
"""
import json, os, glob

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CONTENT_DIR = os.path.join(os.path.dirname(__file__), "content", "the-everglades-cipher")

# ---------------------------------------------------------------------------
# Load chapter summaries / key content for grounding the Q&A
# All Q&A is grounded in actual novel content
# ---------------------------------------------------------------------------

# These are systematically generated Q&A pairs covering all aspects of the novel
# at multiple difficulty levels: recall, analysis, synthesis, evaluation

COMPREHENSIVE_RLHF = [
    # ---- EASY RECALL QUESTIONS ----
    {
        "prompt": "What is the name of Jake Malone's houseboat?",
        "chosen": "Jake Malone's houseboat is called the Reluctant Witness. It is a 1978 Bertram 28 Sport Fisherman moored at the Dinner Key marina in Coconut Grove, Miami. Jake has lived aboard it since 2010.",
        "rejected": "Jake Malone's houseboat is named after a legal term because he used to be a detective. It is docked in Miami somewhere near Coconut Grove."
    },
    {
        "prompt": "Where is Jake Malone's office located?",
        "chosen": "Jake Malone's office is on Tigertail Avenue in Coconut Grove, Miami, above a dry cleaner. The office had previously been a doctor's waiting room and before that a dental practice. It is a second-floor space with a view that Jake finds adequate.",
        "rejected": "Jake Malone has an office in Miami somewhere in Coconut Grove. He works there when he's not on his boat."
    },
    {
        "prompt": "What is the name of Arturo's bookshop?",
        "chosen": "Arturo Vasquez-Cortez's bookshop is called La Luz de los Libros, which translates as 'The Light of Books.' It is located on Calle Ocho (Eighth Street) in Little Havana, Miami, and has been operating since 1978. Arturo has owned the building since 1985.",
        "rejected": "Arturo's bookshop is called something in Spanish that means 'The Light of Books.' It is in Little Havana in Miami."
    },
    {
        "prompt": "What does Elena Vasquez-Cortez do for a living?",
        "chosen": "Elena Vasquez-Cortez is a marine biologist and faculty member at the University of Miami. She specializes in coral reef ecology, specifically studying the interaction between water quality variables and coral recruitment dynamics on the Florida Reef Tract. She has seventeen years of continuous survey data from the Florida Keys.",
        "rejected": "Elena is a scientist who works at a Miami university. She studies ocean ecology and the coral reefs in the Florida Keys."
    },
    {
        "prompt": "What year was the Codex Almeida discovered by Arturo?",
        "chosen": "Arturo Vasquez-Cortez acquired the Codex Almeida in 1989, when it arrived in a box of materials from a Seville estate sale. He identified its significance within the first hour of examination. He spent the following thirty-four years — until 2023 — decoding the cipher system it contained.",
        "rejected": "Arturo found the Codex Almeida in the late 1980s at an estate sale in Spain. He worked on decoding it for many years after that."
    },
    {
        "prompt": "Who is Nacho Reyes and where does he work?",
        "chosen": "Nacho (Ignacio) Reyes is Jake Malone's oldest friend, whom Jake has known since they were twelve years old growing up in Hialeah. He operates a dive charter business from Big Pine Key in the Lower Florida Keys, taking divers to Looe Key reef and other sites. He has been running this business for thirty years and keeps meticulous logs of his dives.",
        "rejected": "Nacho is Jake's old friend from childhood who now lives in the Florida Keys and runs a diving business. He has a boat and knows the Caribbean waters well."
    },
    {
        "prompt": "What is the Order of Christ?",
        "chosen": "The Order of Christ is a Portuguese military-religious order founded in 1319 by King Denis I as the successor organization to the Knights Templar in Portugal. Denis I refused to execute the Templar dissolution decree and established the new Order to absorb the Portuguese Templar personnel, assets, and institutional knowledge — including navigational charts and maritime expertise. The Order of Christ later became the institutional patron of Portuguese maritime exploration in the fifteenth century.",
        "rejected": "The Order of Christ was a Portuguese religious order that replaced the Knights Templar after they were dissolved. It was important in the history of Portuguese exploration."
    },
    {
        "prompt": "What charges were leveled against the Knights Templar during the dissolution?",
        "chosen": "The Knights Templar were charged with heresy: specifically, worshipping an idol called Baphomet, ritual spitting on the cross at initiation ceremonies, engaging in sodomy, denying Christ, and granting absolution without priestly ordination. These charges were not supported by independent evidence and were primarily extracted through torture. Most who confessed subsequently recanted their confessions once the immediate pressure of torture was removed. The Council of Vienne dissolved the Order 'by way of provision' — a legal formula that acknowledged the weakness of the heresy charges.",
        "rejected": "The Templars were accused of heresy and other crimes including worshipping idols and blasphemy. These charges were mostly obtained through torture. The trials were politically motivated by King Philip IV of France."
    },
    {
        "prompt": "What happened to Grand Master Jacques de Molay?",
        "chosen": "Jacques de Molay, the last Grand Master of the Knights Templar, was burned at the stake on March 18, 1314, on an island in the Seine in Paris, along with Geoffroy de Charney. Both had confessed under torture to the heresy charges, recanted their confessions, and were therefore condemned as relapsed heretics. De Molay's last recorded words were a curse on Philip IV of France and Pope Clement V, prophesying that both would answer to God within the year. Both died within fourteen months.",
        "rejected": "Jacques de Molay was the last leader of the Knights Templar. He was executed in Paris in 1314 after being tortured and confessing to heresy. Before he died he cursed the king and the pope who had ordered his death."
    },
    {
        "prompt": "What is a portolan chart?",
        "chosen": "A portolan chart is a class of medieval navigational chart characterized by accurate coastline representation and networks of rhumb lines (lines of constant compass bearing) drawn from multiple compass rose points. They are the most accurate cartographic documents of the medieval period and reflect sophisticated surveying and navigational measurement. The earliest surviving portolan charts date to the late thirteenth century. The Templar Order possessed portolan charts, which contributed to its navigational capabilities. The Cayo Espiritu archive contains twelve navigational charts, including three portolans that the preliminary assessment suggests predate the known tradition by forty years.",
        "rejected": "Portolan charts are a type of medieval map that was used for navigation. They were very accurate for their time and showed coastlines and navigation directions. The Templars had access to these charts which helped with their sailing."
    },
    {
        "prompt": "Who is Victor Andrade?",
        "chosen": "Victor Andrade is the novel's primary antagonist: a sixty-one-year-old Miami real estate developer who learns about Arturo Vasquez-Cortez's Codex research through an academic contact and plans to acquire the Cayo Espiritu site for commercial development before the archive becomes public knowledge. He has Arturo removed to an Everglades fish camp to obtain the coordinates before any government or academic authority can establish a claim. His motivation is purely commercial — the historical content of the archive is irrelevant to him; he wants the land rights.",
        "rejected": "Victor Andrade is the main villain of the novel who wants to use the archive location for a real estate development project. He kidnaps Arturo to get the information he needs. He represents corrupt commercial interests that want to exploit historical discoveries for profit."
    },
    {
        "prompt": "What is Father Benedetto Calvino's role in the novel?",
        "chosen": "Father Benedetto Calvino is a Vatican representative who appears in the novel as a secondary antagonist with a distinct motivation from Andrade's commercial interest. He wants the Philip IV letters in the archive suppressed because they document a relationship between the French crown and the papal Curia that the modern Church would prefer remained obscure. His argument is partly self-interested and partly reasonable: the letters, without careful historical contextualization, could generate damaging media coverage. He represents institutional memory working against historical transparency.",
        "rejected": "Father Calvino is from the Vatican and wants to hide some of the documents found in the archive. He is worried about how the letters from Philip IV will look for the Catholic Church. He tries to prevent the archive from being fully revealed."
    },
    {
        "prompt": "How many documents were found in the Cayo Espiritu archive?",
        "chosen": "The formal excavation of the Cayo Espiritu archive, conducted over the eleven weeks following the initial discovery, revealed 212 documents in total: four letters from Philip IV of France, sixty-four administrative documents of the Iberian Templar preceptories, eighteen theological manuscripts, seven Arabic texts, four Hebrew manuscripts, twelve navigational charts and portolan maps, and eighty-three personal letters and documents of Templar officers. Three of the twelve navigational charts predate the known portolan tradition by forty years.",
        "rejected": "The archive contained over 200 documents including letters from the French king, administrative records, navigation charts, and other manuscripts. There were also some Arabic and Hebrew texts. The most significant items were the letters from Philip IV."
    },
    # ---- MEDIUM ANALYSIS QUESTIONS ----
    {
        "prompt": "Why did Philip IV of France destroy the Knights Templar, according to the novel's historical analysis?",
        "chosen": "The novel proposes that Philip IV's motivations for the Templar dissolution were more complex than the standard historical account of debt-avoidance and anti-clerical politics. The Philip IV letters in the Cayo Espiritu archive document an eleven-year intelligence partnership between the French crown and the Templar Order, beginning in 1296, in which the Templars provided Philip with intelligence on the papal Curia, the English court, and his domestic opponents. The 1306 letter references 'the deterioration of relations' over a financial dispute and Philip's anxiety about whether 'the agreements of the preceding ten years remain in force.' The novel's argument is that the dissolution in 1307 was motivated not only by debt and anti-clerical politics but by Philip's need to destroy an organization that had accumulated dangerous knowledge about his own most sensitive affairs — the Templars had become an intelligence risk. The debt is real; the political program is real; but the intelligence anxiety adds a dimension that makes the dissolution's specific form — the sudden, simultaneous, comprehensively destructive execution — more explicable.",
        "rejected": "The novel suggests Philip IV was motivated by his debts to the Templars and his desire for their assets. The Philip IV letters also show he was worried about the intelligence information the Templars had about him. The combination of financial and security motivations explains why he acted so decisively against them."
    },
    {
        "prompt": "How does the Codex Almeida's cipher system compare to other medieval cryptographic techniques?",
        "chosen": "The Codex Almeida's cipher system is significantly more sophisticated than typical medieval European cryptography and draws on a broader range of intellectual traditions. Medieval European ciphers were primarily simple substitution ciphers — replacing letters with other letters or symbols by a fixed key — which are vulnerable to frequency analysis, a technique that Islamic scholars had systematized by the ninth century. Diego's grille cipher avoids this vulnerability entirely: it is not a substitution cipher but a selection cipher, hiding text within text by position rather than by symbol replacement. The security model is different: instead of relying on key secrecy, it relies on the secrecy of a physical object (the grille template) that can be destroyed without leaving any record. The modal cipher layer adds a second independent security layer using publicly known but non-obviously applicable knowledge (church mode numbers). Together, the two-layer system is more robust than any documented European cryptographic practice of the period and demonstrates familiarity with Islamic cryptographic principles, which is consistent with the Templar Order's documented access to Arabic manuscripts.",
        "rejected": "The Codex Almeida's cipher was more advanced than typical medieval European codes. It used a two-layer system combining a physical grille with a musical cipher. Medieval European ciphers were usually simple letter substitutions. The Templars had access to Islamic mathematical knowledge which may have influenced the design."
    },
    {
        "prompt": "What does Nacho's dive log contribute to the novel's investigation beyond its surface function?",
        "chosen": "Nacho's dive log contributes to the investigation in two ways that the novel distinguishes carefully: its intended function (thirty years of records of dive sites, conditions, and observations for operational purposes) and its unintended function (documentation of the Cayo Espiritu acoustic anomaly that identifies the cave location). This dual significance is the novel's most explicit illustration of its argument about patient long-term record keeping: Nacho kept the log for professional reasons, without any knowledge of what it would eventually prove; the fact that the log is accurate, sustained, and organized is what makes it useful to the investigation. The three entries noting 'a tone from somewhere forward, like a held organ note' at a specific site in the Windward Passage would be worthless if Nacho had not also recorded the date, time, precise GPS coordinates, tide state, and weather conditions for each dive — all the contextual information that allows the entries to be used as primary evidence. The log is thus both a plot device and an argument: good record keeping is valuable in ways that cannot be predicted at the time of record creation.",
        "rejected": "Nacho's dive logs are important because they contain information about the underwater acoustic anomaly that leads to the cave. Without these records they might not have found the right location. The logs also show that Nacho had been near the location many times before without realizing what was there."
    },
    {
        "prompt": "How does the novel use the concept of 'the long game' as a narrative and thematic device?",
        "chosen": "The 'long game' operates at multiple levels simultaneously in the novel. Diego de Almeida plays the longest game: designing a system to preserve an archive across an indefinite future, without knowing when or whether it will be recovered. Arturo plays the thirty-four-year game of decoding the Codex, working in parallel with his bookshop, raising Elena, and the ordinary obligations of a life. Nacho plays the thirty-year game of keeping dive logs that he does not know will be relevant to anything. Elena plays the seventeen-year game of maintaining survey data. All of these are versions of the same act: doing careful, sustained work without guarantee of outcome, trusting that the quality of the work creates value that cannot be predicted in advance. The novel implicitly contrasts this with Andrade's short game — the development deal that requires immediate action, the acquisition before anyone else arrives — and proposes that the long game wins, not by speed or force but by the accumulated weight of accuracy maintained over time. The archive was there for seven centuries. The dive log was there for thirty years. Arturo's decipherment was there after thirty-four. None of them waited to be found; they simply existed, accurately, until someone arrived with the preparation to use them.",
        "rejected": "The novel uses long-term patience and persistence as themes. Diego created the archive to be found centuries later. Arturo spent 34 years decoding the cipher. Nacho kept dive logs for 30 years. These all show how important sustained effort over time can be. The contrast with Andrade's urgency shows that patience often wins in the end."
    },
    {
        "prompt": "What is the significance of the fact that the Cayo Espiritu cave's resonance is in the Phrygian dominant mode?",
        "chosen": "The cave's natural Phrygian dominant resonance is significant at three levels. At the literal level, it provides the self-verification mechanism that Diego built into his cipher: a decoder who arrives at the correct location hears the place produce the same interval that the cipher uses to identify it, which confirms arrival without requiring any additional information. At the structural level, it links the cipher system (human design, intellectual work, the product of deliberate thought) to the physical world (a cave's natural acoustic properties) in a way that suggests the relationship between thought and reality that the novel treats as one of its central concerns: good work discovers what is actually there rather than imposing what the worker imagines. At the thematic level, the Phrygian dominant mode's specific character — its antiquity, its use in flamenco and Middle Eastern music, its quality of compulsive inevitability — matches the novel's emotional register. The place that announces itself in the cipher's own language is also the place that sounds the way the novel feels at its moments of resolution.",
        "rejected": "The cave's Phrygian dominant resonance confirms that explorers have found the right location because it matches the final element of the cipher. This was deliberately planned by Diego as a verification mechanism. It also has thematic significance because the natural world has encoded the same information that Diego chose to encode artificially."
    },
    {
        "prompt": "What does the novel propose about the relationship between institutional power and historical truth?",
        "chosen": "The novel's argument about institutional power and historical truth is embedded in the archive story: the Cayo Espiritu documents were sealed precisely because they constituted a threat to a specific exercise of institutional power — Philip IV's dissolution of the Templar Order on fraudulent charges. The archive survived because one institution (a faction of the Templar Order, specifically Diego's portion of it) understood that preserving the truth was worth the cost of the preservation. The modern phase of the story reproduces the same structure: the Vatican and certain political actors want specific documents suppressed because their content is inconvenient to current institutional interests; the scholars and the novel's protagonists resist this on the grounds that accuracy matters more than convenience. The novel is not naive about this resistance — it acknowledges that the legal and political situation is genuinely complex — but it is clear about the underlying value: historical truth belongs to the historical record, not to the institutions that would prefer to edit it. What makes this claim convincing rather than simply heroic is the specific content of the truth in question: the Philip IV letters document not abstract wrongdoing but a specific form of institutional corruption (using torture-extracted confessions to destroy a rival organization) that the modern institutions implicated would rather not have documented.",
        "rejected": "The novel argues that historical truth should be accessible even when it is inconvenient for powerful institutions. The archive survived because some Templars valued truth over self-preservation. In the present day, the Vatican and others try to suppress certain documents but the protagonists resist this. The novel comes down on the side of historical transparency."
    },
    {
        "prompt": "How does the novel balance the claims of the Cuban diaspora community against Elena's personal history?",
        "chosen": "The novel does not present the Cuban-American community's history as Elena's problem to navigate or resolve. Elena grew up in Little Havana, is fluent in Spanish, understands the community's political culture and its generational arguments, and is — by residence and formation — part of it. But she is also someone whose relationship to Cuba is not the relationship of exile politics: she is from Cuba biologically and has no personal memory of it; she has no political position on the Castro or post-Castro government; she is not carrying the specific emotional weight of the first-generation exiles or their children. Her discovery of her biological family in Havana — Rosa's surviving sister, Elena's biological aunt — is not presented as a political resolution but as a personal one. The aunt in Havana is a seventy-four-year-old woman who lives in Vedado and was Rosa's younger sister. The political context of the U.S.-Cuba relationship is present as background but is not the emotional content of their first video call. The novel's judgment is that Elena's personal history and the community's political history occupy different registers, and that resolving Elena's personal story does not require resolving the political story — and vice versa.",
        "rejected": "Elena's Cuban origins create a connection to the broader Cuban diaspora story but are handled as a personal rather than political matter. Her discovery of biological family in Cuba is emotional and human, not ideological. The novel shows how personal identity and community history can be related without being identical."
    },
    # ---- HARD SYNTHESIS QUESTIONS ----
    {
        "prompt": "Construct the full chain of historical causation from the Templar dissolution to the Portuguese Age of Discovery, as the novel presents it.",
        "chosen": "The novel's historical argument about the Templar dissolution and the Portuguese Age of Discovery proceeds as follows. In 1307, Philip IV of France executes the dissolution of the Knights Templar, motivated by debt, anti-clerical politics, and — the novel adds — intelligence anxiety created by the eleven-year intelligence partnership documented in the Philip IV letters. The dissolution is comprehensive in France and Avignon but incomplete in Portugal: King Denis I refuses to execute the dissolution decree, opens his own investigation that finds no evidence of heresy, and negotiates the establishment of the Order of Christ in 1319 as the Templar successor. The Order of Christ absorbs the Portuguese Templar personnel, assets, and institutional knowledge — including their maritime capabilities, navigational charts, and accumulated Atlantic experience. Diego de Almeida's Caribbean voyages in 1307-1313, if the Codex is authentic, add a specific component to this transmitted knowledge: practical experience of the Atlantic wind system (specifically the northeast trades outbound and the westerlies inbound) that makes systematic Atlantic exploration feasible. This component of navigational knowledge is transmitted within the Order of Christ's institutional memory. In the fifteenth century, the Order of Christ under Henry the Navigator organizes the systematic exploration of the African coast and Atlantic islands. The explorers who benefit from this institutional inheritance — including those who eventually work out the transatlantic route that Columbus will later use — are building on a foundation that includes, among many other elements, the navigational knowledge that the Templar dissolution transferred to the Order of Christ. The dissolution thus paradoxically enables the Age of Discovery it was never designed to enable: by forcing the consolidation of Templar maritime expertise in the one Iberian successor institution that had the political stability and royal patronage to use it systematically.",
        "rejected": "The chain goes: Templars dissolved → Order of Christ created in Portugal → Order inherits Templar knowledge → Order sponsors exploration → Age of Discovery. The Codex Almeida adds the proposal that Diego's Caribbean voyages were part of the knowledge transmitted. This chain connects the 14th century persecution of the Templars to the 15th century Portuguese discoveries."
    },
    {
        "prompt": "How does the novel use parallel structure across centuries to develop its central argument about preservation?",
        "chosen": "The novel's central argument about preservation is constructed through a precise parallel structure across two time periods. In the fourteenth century, Diego de Almeida faces the problem of preserving materials from destruction by a hostile institutional actor (Philip IV). His solution is to remove the materials from the reach of institutional authority entirely — place them in a location (Caribbean sea cave) beyond any jurisdiction, in a form (lead-lined, pitch-coated containers) designed for physical permanence, with a retrieval system (the Codex cipher) designed to work without institutional guardianship, addressed to a future reader he cannot identify. In the twenty-first century, Arturo Vasquez-Cortez faces the problem of preserving knowledge in a different sense: preserving the decipherment work that has consumed thirty-four years of his life, protecting it from the institutional threats that Andrade and Calvino represent, and preserving the personal relationship with Elena by finally providing the knowledge he has been withholding. His solution is similarly to work quietly, in private, completing the task before hostile actors can interfere, and then to transfer what he knows to those who can be trusted to use it responsibly (Elena, Jake, the scholarly community). The parallel extends to the novel's ecological subplot: Elena's preservation of accurate reef survey data against the degradation of the reef itself, and Nacho's preservation of accurate dive records against the possibility that the information they contain will be lost with him. In each case, preservation is not passive but active: it requires sustained effort, accurate practice, and the conviction that the work will matter even when the outcome is not guaranteed.",
        "rejected": "The novel creates parallels between Diego's preservation of the archive in the 14th century and Arturo's preservation of knowledge in the 21st century. Both characters protect important information from hostile actors through careful work and long patience. The ecological subplot provides a third parallel about preserving scientific records. These parallels reinforce the novel's themes about the value of careful, sustained preservation work."
    },
    {
        "prompt": "Evaluate the novel's treatment of the question of what constitutes legitimate scholarly evidence, drawing on specific examples from the text.",
        "chosen": "The novel engages the question of scholarly evidence through three related examples that establish a graduated argument. The Codex Almeida itself is the primary case: Arturo spends thirty-four years developing evidence for his interpretation rather than publishing preliminary findings, because his disciplinary standard requires that claims be supported by evidence sufficient to be compelling before they are made public. He distinguishes consistently between what the document says and what he wants it to say — the methodological discipline that defines good scholarship in any field. When he finally publishes, the evidence is complete. The Philip IV letters are the second case: documentary evidence that moves a historical argument from 'well-supported inference' to 'documented fact.' The letters provide specificity — dates, requests, acknowledgments — that changes the epistemic status of the claim about Philip's intelligence relationship with the Templars. The preliminary assessment of the portolan charts that predate the known tradition is the third case: the novel is careful to present this as a preliminary finding requiring further work, explicitly noting that the implications are large enough to require a higher standard of evidence. Together these examples argue that the appropriate evidentiary standard scales with the magnitude of the claim: thirty-four years of work for the Codex decipherment is appropriate because the claim is large; careful archival authentication for the Philip IV letters is appropriate because the implications are historically significant; extra caution for the portolan charts is appropriate because pre-Columbian European exploration of the Americas is an extraordinary claim that the evidence, however suggestive, has not yet reached a level that justifies asserting.",
        "rejected": "The novel treats scholarly evidence carefully through Arturo's methodological discipline, the authentication of the Philip IV letters, and the cautious approach to the portolan charts. It argues that evidence standards should match the scale of the claims being made. Arturo's 34-year timeline reflects appropriate rigor for a major historical discovery."
    },
    # ---- EXPERT LEVEL QUESTIONS ----
    {
        "prompt": "Analyze how The Everglades Cipher uses the conventions of the thriller genre while transcending them.",
        "chosen": "The novel uses thriller conventions as structural scaffolding that it then populates with content that exceeds the genre's typical ambitions. The conventions it uses: the missing persons case that opens onto something much larger; the ticking clock (Arturo's captivity, Andrade's parallel move to secure the site); the competent investigator who accumulates information through skill and patience; the antagonists with clear motivations; the physical confrontation and rescue sequence; the voyage to the discovery site; the revelation of what was hidden. These are recognizable thriller mechanics, and the novel deploys them competently. What it adds to the conventions: historical depth that treats the fourteenth-century backstory as equally important as the contemporary plot, not as exposition to be consumed and left behind. Character interiority that is more specific than thriller protagonists typically receive — Jake's philosophy of attention, Arturo's methodological discipline, Elena's relationship to both her scientific practice and her personal history. Thematic weight that goes beyond the thriller's usual resolution of threat: the novel is not about preventing bad things from happening to good people (though it is that too); it is about what things are worth preserving and what practices of attention allow preservation to succeed. The ecological subplot runs entirely outside the thriller genre's conventions and serves the novel's thematic purposes without serving its plot. These additions do not make the novel slow or inaccessible; they make it re-readable in the way that novels are re-readable but thrillers usually are not.",
        "rejected": "The novel uses standard thriller elements like the missing person plot, villains with clear motives, and a race to a discovery location. But it transcends the genre through historical depth, complex characters, and serious themes about preservation and scholarship. The ecological subplot also goes beyond typical thriller concerns. The novel is more literary than a standard thriller while still being engaging and plot-driven."
    },
    {
        "prompt": "Compare the characters of Arturo Vasquez-Cortez and Diego de Almeida as the novel constructs them, and explain what this comparison reveals about the novel's central argument.",
        "chosen": "Arturo and Diego are the novel's two great acts of preservation, separated by seven centuries, constructed in deliberate parallel that makes the comparison explicit and available. Both are archivists in the broadest sense: they preserve materials of importance from destruction or loss, with patience and precision, without guarantee of outcome, trusting that the quality of the work creates its own value. Both work in private, without institutional support, because the institutions that might have supported them are either hostile (Philip's France for Diego, the commercial development pressure for Arturo) or simply uninformed (the scholarly community didn't know about the Codex because Arturo didn't tell them). Both address a future reader they cannot identify: Diego's Codex is designed for 'whoever has the preparation to find it'; Arturo's published scholarship is addressed to 'the scholarly community' in the abstract. The parallel is not perfect — Diego was operating under existential threat, in physical danger, executing a complex logistical operation; Arturo was sitting under a lamp in a bookshop for thirty-four years. But the underlying act — trusting that accurate work done carefully will eventually matter, even in the absence of any specific recipient or guarantee — is the same. The comparison reveals the novel's central argument: the practices of careful attention and sustained accuracy constitute a form of action that operates across time in ways that no specific actor can predict or control. Diego could not know Arturo. Arturo could not know Diego. The work connected them anyway.",
        "rejected": "Arturo and Diego are parallel figures across centuries who both preserved important documents through patience and careful work. Diego hid the archive physically and encoded its location; Arturo decoded the location and ensured the archive's scholarly recovery. Their comparison shows that the same kind of careful, dedicated work transcends historical periods and connects people across time. This is the novel's central argument about the value of preservation work."
    },
]

# Load existing RLHF pairs and append without duplicating
existing_path = os.path.join(DATA_DIR, "rlhf_pairs.jsonl")
existing_pairs = []
existing_prompts = set()
with open(existing_path, encoding="utf-8") as f:
    for line in f:
        p = json.loads(line)
        existing_pairs.append(p)
        existing_prompts.add(p["prompt"])

new_pairs = [p for p in COMPREHENSIVE_RLHF if p["prompt"] not in existing_prompts]
all_pairs = existing_pairs + new_pairs

with open(existing_path, "w", encoding="utf-8") as f:
    for pair in all_pairs:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")
print(f"Added {len(new_pairs)} new RLHF pairs")
print(f"Total RLHF pairs: {len(all_pairs)}")
print(f"  Size: {os.path.getsize(existing_path) / 1024:.1f} KB")

# ---------------------------------------------------------------------------
# Also generate a combined "all_training_data.jsonl" in a unified format
# that the notebook can use for any training approach
# ---------------------------------------------------------------------------
combined_path = os.path.join(DATA_DIR, "combined_training_data.jsonl")
all_records = []

# From RLHF pairs
with open(existing_path, encoding="utf-8") as f:
    for line in f:
        pair = json.loads(line)
        # Add as both DPO-style and as instruction-style
        all_records.append({
            "type": "rlhf",
            "prompt": pair["prompt"],
            "chosen": pair["chosen"],
            "rejected": pair["rejected"],
            # instruction version using chosen response
            "instruction": pair["prompt"],
            "input": "",
            "output": pair["chosen"],
        })

# From instruction prompts
instr_path = os.path.join(DATA_DIR, "instruction_prompts.jsonl")
with open(instr_path, encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        all_records.append({
            "type": "instruction",
            "instruction": item["instruction"],
            "input": item.get("input", ""),
            "output": item["output"],
            "prompt": item["instruction"] + ("\n\nContext: " + item["input"] if item.get("input") else ""),
            "chosen": item["output"],
            "rejected": "",
        })

with open(combined_path, "w", encoding="utf-8") as f:
    for record in all_records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
print(f"\nCombined training data: {len(all_records)} records")
print(f"  Size: {os.path.getsize(combined_path) / 1024:.1f} KB")

total = sum(os.path.getsize(os.path.join(DATA_DIR, fn)) for fn in os.listdir(DATA_DIR))
print(f"\nTotal data directory: {total / 1024:.1f} KB ({total / 1024**2:.2f} MB)")
