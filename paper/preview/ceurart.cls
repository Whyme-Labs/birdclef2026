%% ===========================================================================
%% ceurart.cls — LOCAL PREVIEW SHIM (NOT the official CEUR-WS class)
%%
%% This is a minimal stand-in that maps the handful of ceurart macros used by
%% main.tex onto the standard `article` class, so the paper can be compiled for
%% preview on a machine without the real CEURART bundle. It approximates the
%% single-column CEUR look; it is NOT the official template. For the actual
%% submission, obtain the genuine ceurart.cls (Overleaf "CEURART" / CTAN) and
%% compile ../main.tex with it.
%% ===========================================================================
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ceurart}[2026/06/04 v0-preview ceurart shim over article]

\LoadClass[a4paper,10pt]{article}
\RequirePackage[a4paper,margin=2.2cm]{geometry}
\RequirePackage{xkeyval}
\RequirePackage{etoolbox}

% --- \conference{...} : just record it and print as a footnote-ish header line
\newcommand{\conference}[1]{\gdef\@ceurconf{#1}}
\gdef\@ceurconf{}

% --- \author[aff]{name}[keyvals] : accept optional pre- and post- args
% ceurart syntax is \author[1]{Name}[email=..., orcid=...]. We capture name(s).
\let\ceur@origauthor\author
\renewcommand{\author}{\@ifnextchar[{\ceur@author@opt}{\ceur@author@noopt}}
\def\ceur@author@opt[#1]#2{\ceur@author@scan{#2}}
\def\ceur@author@noopt#1{\ceur@author@scan{#1}}
% swallow an optional trailing [keyvals] (email/orcid) if present
\def\ceur@author@scan#1{%
  \g@addto@macro\@ceurauthors{#1\par}%
  \@ifnextchar[{\ceur@author@kv}{}%
}
\def\ceur@author@kv[#1]{} % discard email/orcid for preview
\gdef\@ceurauthors{}

% --- \address[n]{...} : record (printed small under authors)
\newcommand{\address}[2][]{\g@addto@macro\@ceuraddr{#2\par}}
\gdef\@ceuraddr{}

% --- keywords environment with \sep separator
\newenvironment{keywords}{\par\noindent\textbf{Keywords:}\space\itshape}{\par}
\newcommand{\sep}{\textperiodcentered\ }
\newcommand{\keyword}[1]{#1}

% --- redefine \maketitle to lay out title / authors / address / conference
\renewcommand{\maketitle}{%
  \begin{center}
    {\LARGE\bfseries\@title\par}\vspace{1em}
    {\large\@ceurauthors\par}\vspace{0.5em}
    {\small\itshape\@ceuraddr\par}\vspace{0.5em}
    \ifdefempty{\@ceurconf}{}{{\footnotesize\@ceurconf\par}}
  \end{center}\vspace{1em}
}

% abstract already exists in article; keywords printed after it by author order.

% --- acknowledgments environment (ceurart) -> a simple starred section
\newenvironment{acknowledgments}{\section*{Acknowledgments}}{}

% --- CCSXML / printccsdesc are sometimes present; make them no-ops if used
\newenvironment{CCSXML}{\comment}{\endcomment}
\providecommand{\printccsdesc}{}

\endinput
