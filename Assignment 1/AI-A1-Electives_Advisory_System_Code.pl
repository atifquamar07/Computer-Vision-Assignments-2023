:- dynamic ml_courses/1.
:- dynamic math_courses/1.
:- dynamic ece_courses/1.
:- dynamic soc_courses/1.
:- dynamic sec_courses/1.
:- dynamic bio_courses/1.
:- discontiguous form_list_SEC/1.
:- discontiguous form_list_BIO/1.
:- discontiguous form_list_ECE/1.
:- discontiguous form_list_MATH/1.
:- discontiguous form_list_ML/1.
:- discontiguous form_list_SOC/1.
:- style_check(-singleton).

main:-
    write('-------------------------------------------------------------------------------------'), nl,
    write('                WELCOME TO ELECTIVES ADVISORY SYSTEM'), nl,
    write('-------------------------------------------------------------------------------------'), nl, nl, sleep(0.25),
    write('This electives advisory system will suggest you various courses based on your interests and aptitude'), nl,
    write('and also will suggest you possible and good future career choices.'), nl, nl, sleep(0.25),
    write('End each of your answers with a dot (.)'), nl, nl, sleep(0.25),
    
    write('Let us begin'), nl, nl,
    write('Firstly, we will ask you some basic questions regarding your background.'), nl, sleep(0.25),

    read_candidate_name(Name), nl,
    read_candidate_GPA(GPA_ans), nl,

    write('Tell us a bit more about where interest lies in general.'), nl, nl,
    write('Answer the following questions with a (y) or (n).'), nl, nl, sleep(0.25),

    competitive_programming(CP_ans), nl,
    aptitude(AP_ans), nl,
    number_crunching(HFT_ans),

    write('Now we will ask you certain area of fields of study and you have to telll us whether you are'), nl,
    write('interested in exploring the field or not. '), nl, nl,
    write('Answer the following questions with a (y) or (n).'), nl, nl,

    read_ML,
    read_BIO,
    read_MATH,
    read_ECE,
    read_SOC,
    read_SEC,

    write('---------------------------------------------------------------'), nl,
    write('                FUTURE ADVICE FOR YOU'), nl,
    write('---------------------------------------------------------------'), nl, nl, sleep(0.25),

    write(GPA_ans), nl, nl,
    write(AP_ans), nl, nl,
    write(CP_ans), nl, nl,
    write(HFT_ans), nl, nl,
    

    write('---------------------------------------------------------------'), nl,
    write('                RECOMMENDED COURSES FOR YOU'), nl,
    write('---------------------------------------------------------------'), nl, nl, sleep(0.25),


    form_list_ML(ML),
    print_ML_courses(ML),

    form_list_BIO(BIO),
    print_BIO_courses(BIO),

    form_list_MATH(MATH),
    print_MATH_courses(MATH),

    form_list_ECE(ECE),
    print_ECE_courses(ECE),

    form_list_SOC(SOC),
    print_SOC_courses(SOC),

    form_list_SEC(SEC),
    print_SEC_courses(SEC).

   

% -----------------------------------------------------------------------------------
% Background Questions 
% -----------------------------------------------------------------------------------

read_candidate_name(Name):-
    write('Enter your name:- '), nl,
    read(Name), nl, nl,
    write('Hello '), write(Name), write(', once again welcome to THE ELECTIVES ADVISORY SYSTEM !!!!!'), nl,nl.

read_candidate_GPA(GPA_ans):-
    write('What is your current GPA? :- '), nl,
    read(GPA), nl,
    check_GPA(GPA, GPA_ans).


% -----------------------------------------------------------------------------------
% Introductory Questions 
% -----------------------------------------------------------------------------------


competitive_programming(CP_ans) :-
    write('Are you doing competitive programming as hobby?'), nl,
	read(X), nl,
    check_competitive_programming(X, CP_ans).

aptitude(AP_ans) :-
    write('Are you good at data structures and algorithms, problem solving and mental ability?'), nl,
	read(Y), nl,
    check_aptitude(Y, AP_ans).

number_crunching(HFT_ans) :-
    write('Are you good at number crunching, mathematical calculations and algorithms?'), nl,
	read(Z), nl,
    check_number_crunching(Z, HFT_ans).


% -----------------------------------------------------------------------------------
% Checking Introductory Questions 
% -----------------------------------------------------------------------------------


check_competitive_programming('n', CP_ans) :-
	CP_ans = ''.

check_competitive_programming('y', CP_ans) :-
    write('Are you expert rated on Codeforces or a 5-star Coder on CodeChef? '), nl,
    read(Expert), 
    check_expert(Expert, CP_ans).

check_expert('n', CP_ans) :-
    CP_ans = 'Try to imporve your rating on Codeforces and CodeChef as it is a good indicator of how you are in competitive programming.'.

check_expert('y', CP_ans) :-
    CP_ans = 'Excellent!!! You have good problem solving skills and are good in data structures and algorithms. You have a good chance get selected for product based companies as an SDE Role as they want candidates to possess the skillset which you exhibit.'.

check_aptitude('n', AP_ans) :-
	AP_ans = ''.

check_aptitude('y', AP_ans) :-
	AP_ans = 'Great!!! You are good in aptitude. '.

check_number_crunching('n', HFT_ans) :-
	AP_ans = ''.

check_number_crunching('y', HFT_ans) :-
	HFT_ans = 'Wow!!! You are good in number crunching and have good algorithmic skills. You might be a fit for High Frequency Trading companies and Financial Companies.'.

check_GPA(GPA, GPA_ans) :-
    GPA < 8,
	GPA_ans = 'Try improving your GPA. It looks good on your resume, while appyling to a university or a job.'.

check_GPA(GPA, GPA_ans) :-
    GPA >= 8,
	GPA_ans = 'Fantastic!!!, you have maintained your academic performance well!!!'.



% -----------------------------------------------------------------------------------
% Generating List of Recommended courses
% -----------------------------------------------------------------------------------


form_list_ML([Head|Tail]) :-
	retract(ml_courses(Head)),
	form_list_ML(Tail).

form_list_ML([]).


form_list_BIO([Head|Tail]) :-
	retract(bio_courses(Head)),
	form_list_BIO(Tail).

form_list_BIO([]).


form_list_MATH([Head|Tail]) :-
	retract(math_courses(Head)),
	form_list_MATH(Tail).

form_list_MATH([]).


form_list_SOC([Head|Tail]) :-
	retract(soc_courses(Head)),
	form_list_SOC(Tail).

form_list_SOC([]).


form_list_ECE([Head|Tail]) :-
	retract(ece_courses(Head)),
	form_list_ECE(Tail).

form_list_ECE([]).


form_list_SEC([Head|Tail]) :-
	retract(sec_courses(Head)),
	form_list_SEC(Tail).

form_list_SEC([]).



% -----------------------------------------------------------------------------------
% Printing List of Recommended courses
% -----------------------------------------------------------------------------------

print_ML_courses([]).

print_ML_courses([Head|Tail]) :-
	write(Head),nl,
	print_ML_courses(Tail).

print_BIO_courses([]).

print_BIO_courses([Head|Tail]) :-
	write(Head),nl,
	print_BIO_courses(Tail).

print_MATH_courses([]).

print_MATH_courses([Head|Tail]) :-
	write(Head),nl,
	print_MATH_courses(Tail).

print_SOC_courses([]).

print_SOC_courses([Head|Tail]) :-
	write(Head),nl,
	print_SOC_courses(Tail).

print_ECE_courses([]).

print_ECE_courses([Head|Tail]) :-
	write(Head),nl,
	print_ECE_courses(Tail).

print_SEC_courses([]).

print_SEC_courses([Head|Tail]) :-
	write(Head),nl,
	print_SEC_courses(Tail).



% -----------------------------------------------------------------------------------
% Checking Field of interest of the candidate.
% -----------------------------------------------------------------------------------


read_ML:-
    write('Do you wish to explore the paradigm of Machine Learning, Artificial Intelligence and related fields ? :- '), nl, 
    read(ML_ans), nl,
    check_ML(ML_ans).

read_BIO:-
    write('Do you wish to explore the paradigm of BioSciences and Computational Biology ? :- '), nl, 
    read(BIO_ans), nl,
    check_BIO(BIO_ans).

read_MATH:-
    write('Do you wish to explore in the world of Mathematics ? :- '), nl, 
    read(MATH_ans), nl,
    check_MATH(MATH_ans).

read_ECE:-
    write('Do you wish to explore the world of Electronics and Communications ? :- '), nl, 
    read(ECE_ans), nl,
    check_ECE(ECE_ans).

read_SOC:-
    write('Do you wish to explore the world of Social Sciences and Sociology ? :- '), nl, 
    read(SOC_ans), nl,
    check_SOC(SOC_ans).

read_SEC:-
    write('Do you wish to explore the world of Computer Security and how its implemented ? :- '), nl, 
    read(SEC_ans), nl,
    check_SEC(SEC_ans).


% -----------------------------------------------------------------------------------
% Checking Field of interest, are they true or false.
% -----------------------------------------------------------------------------------

check_ML('n') :-
	write('').

check_ML('y') :-
    question_CV, nl,
    question_SML, nl,
    question_DIP, nl,
    question_AI, nl,
    question_NLP, nl.



check_BIO('n') :-
	write('').

check_BIO('y') :-
    question_ACB, nl,
    question_MLBA, nl,
    question_CGAS, nl,
    question_BIP, nl,
    question_BDMH, nl,
    question_ICN, nl.


check_MATH('n') :-
	write('').

check_MATH('y') :-
    question_GT, nl,
    question_CIA, nl,
    question_ANT, nl,
    question_CRN, nl.


check_ECE('n') :-
	write('').

check_ECE('y') :-
    question_DVD, nl,
    question_AELD, nl,
    question_CA, nl,
    question_OWC, nl,
    question_DSP, nl,
    question_BML, nl.
    

check_SOC('n') :-
	write('').

check_SOC('y') :-
    question_SOI, nl,
    question_EI, nl,
    question_STS, nl,
    question_DT, nl,
    question_GMT, nl.

check_SEC('n') :-
	write('').

check_SEC('y') :-
    question_IBC, nl,
    question_TMC, nl.




% -----------------------------------------------------------------------------------
% Suggestion of Courses for Machine Learning Dept.
% -----------------------------------------------------------------------------------

question_CV:-
    write('Are you interested to know how computers and systems to derive meaningful information from digital images,'), nl,
    write('videos and other visual inputs:- '), nl,
    read(CV_ans), nl,
    check_CV_ans(CV_ans).

check_CV_ans('n'):-
    write('').

check_CV_ans('y'):-
    write('Have you done MTH100 : LINEAR ALGEBRA and CSE101 : INTRODUCTION TO PROGRAMMING courses?'), nl,
    read(Pre_req_CV), nl,
    final_check_CV(Pre_req_CV).
    

final_check_CV('y'):-
    assert(ml_courses('CSE 344/544, ECE 344/544 :- COMPUTER VISION')).

final_check_CV('n'):-
    write('').

question_SML:-
    write('Are you interested in studying the theoretical aspects of the design and analysis of machine learning '), nl,
    write('algorithms using tools of statistics and computer science:- '), nl,
    read(SML_ans), nl,
    check_SML_ans(SML_ans).

check_SML_ans('n'):-
    write('').

check_SML_ans('y'):-
    write('Have you done MTH201 - PROBABILTY AND STATISTICS course?'), nl,
    read(Pre_req_SML), nl,
    final_check_SML(Pre_req_SML).


final_check_SML('y'):-
    assert(ml_courses('CSE342/CSE542/ECE356/ECE556 :- STATISTICAL MACHINE LEARNING')).

final_check_SML('n'):-
    write('').



question_DIP:-
    write('Are you interested to study the fundamental theories and algorithms of digital image acquisition,'), nl,
    write('color representation, sampling and quantization? '), nl,
    read(DIP_ans), nl,
    check_DIP_ans(DIP_ans).

check_DIP_ans('n'):-
    write('').

check_DIP_ans('y'):-
    write('Have you done MTH100 - LINEAR ALGEBRA and CSE101 and MTH201 - PROBABILTY AND STATISTICS courses?'), nl,
    read(Pre_req_DIP), nl,
    final_check_DIP(Pre_req_DIP).


final_check_DIP('y'):-
    assert(ml_courses('CSE340/CSE540/ECE340 :- DIGITAL IMAGE PROCESSING')).

final_check_DIP('n'):-
    write('').



question_AI:-
    write('Are you interested to study various search techniques, knowledge representation, reasoning, and learning'), nl,
    write('in the field of Artificial Intelligence? '), nl,
    read(AI_ans), nl,
    check_AI_ans(AI_ans).

check_AI_ans('n'):-
    write('').

check_AI_ans('y'):-
    write('Have you done CSE102 - DATA STRUCTURES AND ALGORITHMS course?'), nl,
    read(Pre_req_AI), nl,
    final_check_AI(Pre_req_AI).


final_check_AI('y'):-
    assert(ml_courses('CSE643 :- ARTIFICIAL INTELLIGENCE')).

final_check_AI('n'):-
    write('').



question_NLP:-
    write('Are you interested in building machines and that understand and respond to text or voice data and '), nl,
    write('respond with text or speech of their own in much the same way humans do? '), nl,
    read(NLP_ans), nl,
    check_NLP_ans(NLP_ans).

check_NLP_ans('n'):-
    write('').

check_NLP_ans('y'):-
    write('Have you done CSE101 : INTRODUCTION TO PROGRAMMING, CSE22 : ANALYSIS AND DESIGN OF ALGORITHMS, MTH100 : LINEAR ALGEBRA, MTH201 : PROBABILTY AND STATISTICS courses?'), nl,
    read(Pre_req_NLP), nl,
    final_check_NLP(Pre_req_NLP).


final_check_NLP('y'):-
    assert(ml_courses('CSE556 :- NATURAL LANGUAGE PROCESSING')).

final_check_NLP('n'):-
    write('').



% -----------------------------------------------------------------------------------
% Suggestion of Courses for BioSciences Dept.
% -----------------------------------------------------------------------------------

question_ACB:-
    write('Are you interested in solving real life biological problems and study basics of algorithm '), nl,
    write('designing techniques and their application in solving problems of molecular biology? '), nl,
    read(ACB_ans), nl,
    check_ACB_ans(ACB_ans).

check_ACB_ans('n'):-
    write('').

check_ACB_ans('y'):-
    assert(bio_courses('BIO522 :- ALGORITHMS IN COMPUTATIONAL BIOLOGY')).


question_MLBA:-
    write('Are you interested in studying machine learning techniques for developing prediction models and'), nl,
    write('solving biomedical problems using machine learning techniques? '), nl,
    read(MLBA_ans), nl,
    check_MLBA_ans(MLBA_ans).

check_MLBA_ans('n'):-
    write('').

check_MLBA_ans('y'):-
    assert(bio_courses('BIO542 :- MACHINE LEARNING IN BIOMEDICAL APPLICATIONS')).



question_CGAS:-
    write('Are you interested in analysis and visualization of culinary data using statistical analysis, text mining'), nl,
    write('natural language processing, machine learning? '), nl,
    read(CGAS_ans), nl,
    check_CGAS_ans(CGAS_ans).

check_CGAS_ans('n'):-
    write('').

check_CGAS_ans('y'):-
    assert(bio_courses('BIO544 :- COMPUTATIONAL GASTRONOMY')).



question_BIP:-
    write('Are you interested in studying physical, engineering and signal processing principles needed for '), nl,
    write('medical imaging and image processing? '), nl,
    read(BIP_ans), nl,
    check_BIP_ans(BIP_ans).

check_BIP_ans('n'):-
    write('').

check_BIP_ans('y'):-
    assert(bio_courses('BIO524 :- BIOMEDICAL IMAGE PROCESSING')).



question_BDMH:-
    write('Are you interested in studying implementation of data mining techniques in healthcare, to solve '), nl,
    write('health-related problems? '), nl,
    read(BDMH_ans), nl,
    check_BDMH_ans(BDMH_ans).

check_BDMH_ans('n'):-
    write('').

check_BDMH_ans('y'):-
    assert(bio_courses('BIO543 :- BIG DATA MINING IN HEALTHCARE')).



question_ICN:-
    write('Are you interested in studying the behaviour of neuronal systems and their respective mathematical models'), nl,
    write('and explore the field of neurosciences? '), nl,
    read(ICN_ans), nl,
    check_ACB_ans(ICN_ans).

check_ICN_ans('n'):-
    write('').

check_ICN_ans('y'):-
    assert(bio_courses('BIO534 :- INTRODUCTION TO COMPUTATIONAL NEUROSCIENCE')).


% -----------------------------------------------------------------------------------
% Suggestion of Courses for Mathematics Dept.
% -----------------------------------------------------------------------------------

question_GT:-
    write('Are you interested in studying concepts related to theory of graphs?  '), nl,
    read(GT_ans), nl,
    check_GT_ans(GT_ans).

check_GT_ans('n'):-
    write('').

check_GT_ans('y'):-
    assert(math_courses('MTH310 :- GRAPH THEORY')).



question_CIA:-
    write('Are you interested in studying combinatorics topics such as permutation groups, linear codes,'), nl,
    write('Stirling and Bell numbers? '), nl,
    read(CIA_ans), nl,
    check_CIA_ans(CIA_ans).

check_CIA_ans('n'):-
    write('').

check_CIA_ans('y'):-
    write('Have you done MTH100 : LINEAR ALGEBRA course?'), nl,
    read(Pre_req_CIA), nl,
    final_check_CIA(Pre_req_CIA).


final_check_CIA('y'):-
    assert(math_courses('MTH311 :- COMBINATORICS AND ITS APPLICATIONS')).

final_check_CIA('n'):-
    write('').



question_ANT:-
    write('Are you interested in studying topics such as number fields, rings of integers, factorization'), nl,
    write('in Dedekind domains, class numbers and class groups, units in rings of integers , valuations'), nl,
    write('and local fields in the field of algebra? '), nl,
    
    read(ANT_ans), nl,
    check_ANT_ans(ANT_ans).

check_ANT_ans('n'):-
    write('').

check_ANT_ans('y'):-
    assert(math_courses('MTH512 :- ALGEBRAIC NUMBER THEORY')).



question_CRN:-
    write('Are you interested in studying Calculas in the nth dimension and topics such as '), nl,
    write('Rham cohomology of manifolds, and to physics via modeling of Maxwells equations? '), nl,
    read(CRN_ans), nl,
    check_CRN_ans(CRN_ans).

check_CRN_ans('n'):-
    write('').

check_CRN_ans('y'):-
    write('Have you done MTH100 : LINEAR ALGEBRA, MTH203 : MULTIVARIATE CALCULUS, MTH204 : REAL ANALYSIS - I courses?'), nl,
    read(Pre_req_CRN), nl,
    final_check_CRN(Pre_req_CRN).


final_check_CRN('y'):-
    assert(math_courses('MTH544 :- CALCULUS IN R(n)')).

final_check_CRN('n'):-
    write('').



% -----------------------------------------------------------------------------------
% Suggestion of Courses for Electronics and Communications Dept.
% -----------------------------------------------------------------------------------

question_DVD:-
    write('Are you interested in knowing how your computer chip works, how ICs work and microelectronics?'), nl,
    write('Also do you want to build a project using state-of-the-art computer aided design (CAD) tools in VLSI? '), nl,
    read(DVD_ans), nl,
    check_DVD_ans(DVD_ans).

check_DVD_ans('n'):-
    write('').

check_DVD_ans('y'):-
    assert(ece_courses('ECE314/ECE514 :- DIGITAL VLSI DESIGN')).




question_AELD:-
    write('Are you interested in learning various embedded system concepts and hardware software co-design approach'), nl,
    write('and how to implement the embedded systems on heterogenos SoC and optimize their area and power performance? '), nl,
    read(AELD_ans), nl,
    check_AELD_ans(AELD_ans).

check_AELD_ans('n'):-
    write('').

check_AELD_ans('y'):-
    assert(ece_courses('ECE573 :- ADVANCED EMBEDDED LOGIC DESIGN')).



question_CA:-
    write('Are you interested in exploring the techniques that go into designing a modern microprocessor,'), nl,
    write('also learn architectural techniques such as multi-issue superscalar processors, out-of-order'), nl,
    write('processors, Very Long Instruction Word (VLIW) processors, advanced caching, and multiprocessor systems?'), nl,
    read(CA_ans), nl,
    check_CA_ans(CA_ans).

check_CA_ans('n'):-
    write('').

check_CA_ans('y'):-
    assert(ece_courses('CSE511/ECE511 :- COMPUTER ARCHITECTURE')).



question_OWC:-
    write('Are you interested in studying Optical Wireless Communications where you will get to learn about the'), nl,
    write(' technology for supporting high-data-rate 5G communication and its massive connectivity of IoT? '), nl,
    read(OWC_ans), nl,
    check_OWC_ans(OWC_ans).

check_OWC_ans('n'):-
    write('').

check_OWC_ans('y'):-
    assert(ece_courses('ECE546 :- OPTICAL AND WIRELESS CONVERGENCE FOR BEYOND 5G NETWORKS AND IOT DEVICES')).



question_DSP:-
    write('Are you interested in studying the theory of digital signal processing where you will concentrating '), nl,
    write('on signal analysis using Fourier transforms, linear system analysis, Filter design and a few more advanced topics? '), nl,
    read(DSP_ans), nl,
    check_DSP_ans(DSP_ans).

check_DSP_ans('n'):-
    write('').

check_DSP_ans('y'):-
    write('Have you done ECE250 : SIGNALS AND SYSTEMS course?'), nl,
    read(Pre_req_DSP), nl,
    final_check_DSP(Pre_req_DSP).

final_check_DSP('n'):-
    write('').

final_check_DSP('y'):-
    assert(ece_courses('ECE351 :- DIGITAL SIGNAL PROCESSING')).






question_BML:-
    write('Are you interested in studying the Bayesian statistical modeling in the context of machine learning, '), nl,
    write('also interesting in setting up a machine learning problem as a Bayesian model and design exact or '), nl,
    write('approximate (sampling/optimization) solution methods for computationally scalable inference?'), nl,
    read(BML_ans), nl,
    check_BML_ans(BML_ans).

check_BML_ans('n'):-
    write('').

check_BML_ans('y'):-
    write('Have you done MTH201 : PROBABILTY AND STATISTICS course?'), nl,
    read(Pre_req_BML), nl,
    final_check_BML(Pre_req_BML).

final_check_BML('n'):-
    write('').

final_check_BML('y'):-
    assert(ece_courses('CSE515/ECE551 :- BAYESIAN MACHINE LEARNING')).

    



% -----------------------------------------------------------------------------------
% Suggestion of Courses for Social Sciences Dept.
% -----------------------------------------------------------------------------------

question_SOI:-
    write('Are you interested in studying the foundations, contemporary debates and relevant themes '), nl,
    write('of Sociology and Social Anthropology in India?'), nl,
    read(SOI_ans), nl,
    check_SOI_ans(SOI_ans).

check_SOI_ans('n'):-
    write('').

check_SOI_ans('y'):-
    assert(soc_courses('SOC313/SOC503 :- SOCIOLOGY OF INDIA : THEMES AND PERSPECTIVES')).



question_EI:-
    write('Are you interested in studying the ethics of AI where you study the questions of fairness, '), nl,
    write('transparency, justice linked to AI and the ethics of its deployment in various '), nl,
    write('settings of healthcare, warfare, autonomous vehicles, education etc? '), nl,
    read(EI_ans), nl,
    check_EI_ans(EI_ans).

check_EI_ans('n'):-
    write('').

check_EI_ans('y'):-
    assert(soc_courses('SSH325/SSH525 :- ETHICS IN AI')).


question_STS:-
    write('Are you interested in studying sociology of science and technology also exploring more fundamental '), nl,
    write('ideas of technology and its relation to science and society, the tutorials and evaluation will be '), nl,
    write('based on case studies from Biotechnology? '), nl,
    read(STS_ans), nl,
    check_STS_ans(STS_ans).

check_STS_ans('n'):-
    write('').

check_STS_ans('y'):-
    assert(soc_courses('SOC211 :- SCIENCE TECHNOLOGY AND SOCIETY')).



question_DT:-
    write('Are you interested in studying foundations for rational decision making in certain and uncertain environments.'), nl,
    write('also the theory of rational choice permeates every discipline that touches upon human behavior, '), nl,
    write('more formally Decision Theory? '), nl,
    
    read(DT_ans), nl,
    check_DT_ans(DT_ans).

check_DT_ans('n'):-
    write('').

check_DT_ans('y'):-
    assert(soc_courses('ECO503 :- DECISION THEORY')).



question_GMT:-
    write('Are you interested in studying Game theoretic modeling and strategic analysis as a distinct '), nl,
    write('methodology has been a major intellectual achievement of the past century not only within '), nl,
    write('the Economics discipline but more broadly in the Social Sciences?'), nl,
    read(GMT_ans), nl,
    check_GMT_ans(GMT_ans).

check_GMT_ans('n'):-
    write('').

check_GMT_ans('y'):-
    assert(soc_courses('ECO311/ECO511 :- GAME THEORY')).



% -----------------------------------------------------------------------------------
% Suggestion of Courses for Computer Security Dept.
% -----------------------------------------------------------------------------------


question_IBC:-

    assert(sec_courses('CSE345/CSE545 :- FOUNDATIONS OF COMPUTER SECURITY')),
    
    write('Are you interested in studying the basics of blockchains and cryptocurrency and how cryptocurrency '), nl,
    write('such as Bitcoin and Ethereum work, blockchain and other decentralized consensus protocols, digital coin mining? '), nl,
    read(IBC_ans), nl,
    check_IBC_ans(IBC_ans).

check_IBC_ans('n'):-
    write('').

check_IBC_ans('y'):-
    write('Have you done CSE546 : APPLIED CRYPTOGRAPHY course?'), nl,
    read(Pre_req_IBC), nl,
    final_check_IBC(Pre_req_IBC).

final_check_IBC('n'):-
    write('').

final_check_IBC('y'):-
    assert(sec_courses('CSE528 :- INTRODUCTION TO BLOCKCHAIN AND CRYPTOGRAPHY')).



question_TMC:-
    write('Are you interested in studying the principles of Modern Cryptography and learn '), nl,
    write('learn how modern cryptography works in real life?'), nl,
    read(TMC_ans), nl,
    check_TMC_ans(TMC_ans).

check_TMC_ans('n'):-
    write('').

check_TMC_ans('y'):-
    assert(sec_courses('CSE524 :- THEORY OF MODERN CRYPTOGRAPHY')).