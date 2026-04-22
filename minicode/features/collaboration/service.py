from __future__ import annotations


class CollaborationService:
    def __init__(self, repository: object) -> None:
        self.repository = repository

    def register_agent(self, name: str, description: str, skills: list[str]) -> None:
        self.repository.register_agent(name, description, skills)

    def list_agents(self):
        return self.repository.list_agents()

    def post_task(self, channel: str, sender: str, recipient: str, content: str) -> None:
        self.repository.post_message(channel, sender, recipient, content)

    def route_messages(self, channel: str):
        return self.repository.list_messages(channel)

    def format_status(self) -> str:
        agents = self.repository.list_agents()
        if not agents:
            return "No agents registered."
        return "\n".join(f"{agent['name']}: {agent['status']} ({', '.join(agent['skills'])})" for agent in agents)
