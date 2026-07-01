import { useEffect, useState } from "react";
import {
  ArrowRight,
  Briefcase,
  CheckCircle,
  ClipboardText,
  Code,
  Copy,
  DownloadSimple,
  EnvelopeSimple,
  FileCode,
  GraduationCap,
  GithubLogo,
  List,
  LinkedinLogo,
  ShieldCheck,
  Stack,
  TerminalWindow,
  X,
} from "@phosphor-icons/react";

import heroDiagram from "./assets/hero-systems-architecture.png";
import reliabilityDiagram from "./assets/realtime-reliability.png";

const EMAIL = "CarnlineJonathan@gmail.com";
const SOURCE_URL = "https://github.com/Miste-r-J/jonathan-carnline-portfolio/tree/main/projects/reliable-trading-runtime";
const assetPath = (path) => `${import.meta.env.BASE_URL}${path.replace(/^\/+/, "")}`;

const projects = [
  {
    number: "01",
    id: "reliability",
    title: "Reliable Trading Runtime",
    subtitle: "The real Python, C#, PowerShell, test, and operations source",
    image: reliabilityDiagram,
    problem:
      "I needed to know what the software actually did when market data, model decisions, safety rules, the order bridge, and broker state did not agree.",
    work:
      "I built a complete runtime for training, backtesting, replay, paper, and live operation. It calculates features, evaluates models, governs decisions, sends canonical order intents through a C# NinjaTrader bridge, reconciles fills and positions, and records the full evidence chain.",
    verification:
      "I backed it with 76 test modules plus replay, parity, audit, and diagnostic tools. Signals, gates, intents, acknowledgements, fills, positions, protection state, health, and PnL remain separate and traceable.",
    stack: ["Python", "C#", "PowerShell", "pandas", "scikit-learn", "pytest", "NinjaTrader 8"],
    notes: [
      "I separate prediction, permission, intent, acknowledgement, and fill so one status cannot hide where a failure occurred.",
      "I use stable correlation IDs, deduplication, append-only ledgers, atomic writes, broker snapshots, and conservative recovery rules.",
      "The repository contains the real implementation and tests. Only credentials, account data, raw logs, datasets, binaries, and generated artifacts are excluded.",
    ],
    proof: {
      title: "Review the actual source",
      summary:
        "I explain the design in first person, provide a file-by-file starting point, and publish the source instead of a marketing example.",
      facts: ["256 Python files", "About 117,000 lines", "76 test modules", "Python/C#/PowerShell"],
      download: SOURCE_URL,
    },
  },
];

const experience = [
  {
    dates: "2024 — Present",
    company: "The Home Depot",
    role: "New Associate Coach / Warehouse Associate",
    detail: "Selected for the Voice of Associates team and promoted within four months; train and support new hires in safe, efficient distribution operations.",
  },
  {
    dates: "2024",
    company: "Walmart Distribution Center",
    role: "Replenishment Associate",
    detail: "Maintained stock flow and coordinated replenishment across continuous warehouse operations.",
  },
  {
    dates: "2021 — 2023",
    company: "Target Distribution Center",
    role: "Team Member Trainer / Warehouse Worker",
    detail: "Advanced from freight unloading to equipment operation and employee training in a high-volume receiving environment.",
  },
  {
    dates: "2019 — 2021",
    company: "Shakey T’s Polish & Installations",
    role: "Office Manager / Electrical Installer",
    detail: "Progressed from shop-floor work into electrical installation, technical troubleshooting, scheduling, inventory, and office operations.",
  },
];

const stackGroups = [
  { icon: Code, label: "Languages", items: "Python · C# · PowerShell · PHP · SQL · HTML/CSS" },
  { icon: Stack, label: "Data & ML", items: "pandas · NumPy · XGBoost · SQLite · CSV/JSONL pipelines" },
  { icon: TerminalWindow, label: "Systems", items: "Windows · Linux · WSL · Docker · systemd · SSH/SFTP" },
  { icon: ShieldCheck, label: "Reliability", items: "Idempotency · Reconciliation · Observability · Fail-closed controls" },
];

function Header() {
  const [open, setOpen] = useState(false);
  const close = () => setOpen(false);

  return (
    <header className="site-header">
      <a className="brand" href="#top" aria-label="Jonathan Carnline home">
        <span>JONATHAN CARNLINE</span>
        <small>Operations. Automation. Reliability.</small>
      </a>
      <button className="menu-button" onClick={() => setOpen((value) => !value)} aria-label="Toggle navigation" aria-expanded={open}>
        {open ? <X size={24} /> : <List size={24} />}
      </button>
      <nav className={open ? "site-nav open" : "site-nav"} aria-label="Primary navigation">
        <a href="#work" onClick={close}>Work</a>
        <a href="#leadership" onClick={close}>Leadership</a>
        <a href="#stack" onClick={close}>Stack</a>
        <a href="#about" onClick={close}>About</a>
        <a href="#contact" onClick={close}>Contact</a>
        <a className="resume-link" href={assetPath("Jonathan_Carnline_Technical_Resume.pdf")} download onClick={close}>
          Download résumé <DownloadSimple size={17} weight="bold" />
        </a>
      </nav>
    </header>
  );
}

function Project({ project }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <article className="project-detail" id={project.id}>
      <div className="case-copy">
        <div className="case-title-row">
          <span className="case-number">{project.number}</span>
          <div>
            <h3>{project.title}</h3>
            <p className="case-subtitle">{project.subtitle}</p>
          </div>
        </div>
        <dl className="case-facts">
          <div><dt>Problem</dt><dd>{project.problem}</dd></div>
          <div><dt>Work</dt><dd>{project.work}</dd></div>
          <div><dt>Verification</dt><dd>{project.verification}</dd></div>
        </dl>
        <div className="tags" aria-label={`${project.title} technologies`}>
          {project.stack.map((item) => <span key={item}>{item}</span>)}
        </div>
        {project.proof && (
          <aside className="proof-card" aria-label={`${project.title} public proof`}>
            <div>
              <FileCode size={22} weight="duotone" />
              <div>
                <h4>{project.proof.title}</h4>
                <p>{project.proof.summary}</p>
              </div>
            </div>
            <ul>{project.proof.facts.map((fact) => <li key={fact}>{fact}</li>)}</ul>
            <a href={project.proof.download} target="_blank" rel="noreferrer">Open source on GitHub <GithubLogo size={16} weight="bold" /></a>
          </aside>
        )}
        <button className="text-button" onClick={() => setExpanded((value) => !value)} aria-expanded={expanded}>
          {expanded ? "Close technical notes" : "Open technical notes"} <ArrowRight size={16} weight="bold" />
        </button>
        {expanded && (
          <div className="technical-notes">
            <h4>Technical notes</h4>
            <ul>{project.notes.map((note) => <li key={note}><CheckCircle size={18} weight="fill" /> <span>{note}</span></li>)}</ul>
          </div>
        )}
      </div>
      <figure className="case-figure">
        <img src={project.image} alt={`Architecture illustration for ${project.title}`} />
      </figure>
    </article>
  );
}

export function App() {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const sections = document.querySelectorAll("section[id]");
    const observer = new IntersectionObserver(
      (entries) => entries.forEach((entry) => entry.target.classList.toggle("in-view", entry.isIntersecting)),
      { threshold: 0.08 },
    );
    sections.forEach((section) => observer.observe(section));
    return () => observer.disconnect();
  }, []);

  const copyEmail = async () => {
    try {
      if (!navigator.clipboard?.writeText) throw new Error("Clipboard API unavailable");
      await navigator.clipboard.writeText(EMAIL);
    } catch {
      const field = document.createElement("textarea");
      field.value = EMAIL;
      field.setAttribute("readonly", "");
      field.style.position = "fixed";
      field.style.opacity = "0";
      document.body.appendChild(field);
      field.select();
      document.execCommand("copy");
      document.body.removeChild(field);
    }
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  };

  return (
    <>
      <Header />
      <main id="top">
        <section className="hero in-view" aria-labelledby="hero-title">
          <div className="hero-copy">
            <p className="eyebrow">Operations, automation, and reliability</p>
            <h1 id="hero-title">Jonathan Carnline</h1>
            <p className="hero-statement">I build reliable tools where software meets real operations.</p>
            <p className="hero-summary">
              I like work where the details matter: clear status, safe controls, clean handoffs, and proof that the work actually ran.
            </p>
            <div className="hero-actions">
              <a className="button primary" href="#work">Read the work <ArrowRight size={18} weight="bold" /></a>
              <a className="button secondary" href={assetPath("Jonathan_Carnline_Technical_Resume.pdf")} download>
                Download résumé <DownloadSimple size={18} weight="bold" />
              </a>
            </div>
          </div>
          <figure className="hero-figure">
            <img src={heroDiagram} alt="Connected applications, automation, observability, and data tools" />
            <figcaption>Build it clearly. Prove it works.</figcaption>
          </figure>
        </section>

        <section className="project-index" aria-labelledby="project-index-title">
          <div className="section-label"><ClipboardText size={18} /><span id="project-index-title">Project index</span></div>
          <div className="index-grid">
            {projects.map((project) => (
              <a href={`#${project.id}`} key={project.id}>
                <span>{project.number}</span>
                <strong>{project.title}</strong>
                <small>{project.subtitle}</small>
              </a>
            ))}
          </div>
        </section>

        <section className="work-section" id="work" aria-labelledby="work-title">
          <div className="section-heading">
            <p className="eyebrow">Selected engineering work</p>
            <h2 id="work-title">The system I built.</h2>
            <p>This is the actual source, how I built it, what I owned, and how I verified the difficult parts.</p>
          </div>
          {projects.map((project) => <Project project={project} key={project.id} />)}
        </section>

        <section className="stack-section" id="stack" aria-labelledby="stack-title">
          <div className="section-label"><Stack size={18} /><span>Technical stack</span></div>
          <div className="stack-intro">
            <h2 id="stack-title">Broad enough to integrate. Focused enough to debug.</h2>
            <p>My strongest work sits where code, runtime behavior, Windows/Linux, and real users meet.</p>
          </div>
          <div className="stack-grid">
            {stackGroups.map(({ icon: Icon, label, items }) => (
              <div className="stack-group" key={label}>
                <Icon size={24} weight="duotone" />
                <h3>{label}</h3>
                <p>{items}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="leadership-section" id="leadership" aria-labelledby="leadership-title">
          <div className="section-label"><Briefcase size={18} /><span>Operations leadership</span></div>
          <div className="section-heading compact">
            <h2 id="leadership-title">A consistent pattern of learning, ownership, and training.</h2>
            <p>The technical work is independent project work. The professional record below shows the operational habits behind it.</p>
          </div>
          <div className="timeline">
            {experience.map((item) => (
              <article key={`${item.company}-${item.dates}`}>
                <span className="timeline-date">{item.dates}</span>
                <h3>{item.company}</h3>
                <h4>{item.role}</h4>
                <p>{item.detail}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="about-section" id="about" aria-labelledby="about-title">
          <div className="about-card">
            <GraduationCap size={27} weight="duotone" />
            <p className="eyebrow">Education</p>
            <h2 id="about-title">Computer Information Systems coursework</h2>
            <p>Bellevue University — online · 30 semester credits completed toward a Bachelor of Science.</p>
          </div>
          <div className="about-card">
            <ShieldCheck size={27} weight="duotone" />
            <p className="eyebrow">Working style</p>
            <h2>Evidence-led and safety-minded</h2>
            <p>I do not call something healthy just because it says connected. I check the actual log, state change, run output, or proof file.</p>
          </div>
        </section>

        <section className="contact-section" id="contact" aria-labelledby="contact-title">
          <div>
            <p className="eyebrow">Let’s connect</p>
            <h2 id="contact-title">Building reliable tools that teams can trust.</h2>
            <p>Open to opportunities in application support, operations technology, production support, Python/C# work, DevOps, logistics technology, and automation.</p>
          </div>
          <div className="contact-actions">
            <a className="button primary" href={`mailto:${EMAIL}`}><EnvelopeSimple size={19} weight="bold" /> Email Jonathan</a>
            <a className="button secondary" href="https://www.linkedin.com/in/jonathan-carnline-b70420262" target="_blank" rel="noreferrer"><LinkedinLogo size={19} weight="bold" /> LinkedIn</a>
            <button className="button secondary" onClick={copyEmail}><Copy size={19} weight="bold" /> {copied ? "Email copied" : "Copy email address"}</button>
          </div>
        </section>
      </main>
      <footer>
        <span>© {new Date().getFullYear()} Jonathan Carnline</span>
        <a href="#top">Back to top <ArrowRight size={15} weight="bold" /></a>
      </footer>
    </>
  );
}
