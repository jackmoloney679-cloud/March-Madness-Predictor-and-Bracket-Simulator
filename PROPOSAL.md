# March Madness Predictor and Bracket Simulator  
**Author:** Jack Moloney  
**Category:** Sports Statistical Analysis / Monte Carlo Simulation and Modeling  

---

### **Problem Statement & Motivation**
During the month of March, millions of Americans attempt a fascinating yet almost impossible challenge: creating the perfect NCAA March Madness bracket. Basketball fans across the country try to predict the winners of 63 tournament games. Factors such as the single-elimination format, frequent upsets (lower-seed teams defeating higher ones), and unlikely “Cinderella” stories make this challenge extremely hard to achieve.

This project aims to explore why predicting March Madness is nearly impossible by building a Python-based model that analyzes past tournament data and simulates future outcomes. The goal is not to find the perfect bracket, but to better understand the probabilistic nature of the event and visualize how different factors — like team seeding or efficiency — affect outcomes.

---

### **Planned Approach & Technologies**
The project will use **Python 3.10+** as the main language, with libraries including **pandas**, **NumPy**, **matplotlib**, and **pylint** for code quality checks.  
Other tools or languages may be added later as needed.  

Data sources will include:
- Kaggle’s *March Machine Learning Mania* datasets  
- FiveThirtyEight’s NCAA tournament prediction repository  
- (Optional) Sports Reference College Basketball data (team/player stats such as offensive and defensive efficiency, strength of schedule)  
- (Optional) NCAA dataset of historical results and box scores (1894–2018)

---

### **Proposed Steps**
1. Clean all historical tournament data (teams, seeds, results)  
2. Analyze and interpret data (win rate per seed, upset frequency)  
3. Build a regression or random forest model to predict win probabilities for any matchup  
4. Run **Monte Carlo simulations** to generate many possible outcomes  
5. Visualize results such as the probability of each team reaching specific stages (Sweet 16, Elite 8, Final Four, National Championship)

---

### **Expected Challenges**
- Managing and merging data across multiple tournaments  
- Representing uncertainty and randomness in simulations  
- Verifying efficiency and accuracy of simulation results  
- Avoiding overcrowding or duplication in the dataset  

---

### **Success Criteria**
The project will be considered successful if it produces:
- A working simulation model that generates realistic game outcomes  
- Visualizations showing team progression and advancement probabilities  
- A clear explanation of why March Madness is so unpredictable  

---

### **Stretch Goals**
If time permits:
- Compare predictions with FiveThirtyEight’s or ESPN’s models  
- Extend the simulation to include the **Women’s NCAA Tournament**  

---
