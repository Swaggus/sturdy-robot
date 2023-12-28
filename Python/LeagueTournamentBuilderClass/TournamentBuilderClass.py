import random

class tournament:
    def __init__(self, tName, id, duration_days, prize_pool, game_type):
        self.tName = tName
        self.id = id
        self.duration_days = duration_days
        self.prize_pool = prize_pool
        self.game_type = game_type
        self.teamRankings = [{'team':'test', 'score':0, 'wins':0, 'losses':0, 'draws':0}]
    
    def create_match_ups(self, ActiveTeamList):
        self.ActiveTeamList = ActiveTeamList
        self.match_ups = []
        for i in range(0, len(ActiveTeamList), 2):
            self.match_ups.append([ActiveTeamList[i], ActiveTeamList[i + 1]])
        return self.match_ups
    
    def initialise_team_rankings(self, ActiveTeamList):
        self.ActiveTeamList = ActiveTeamList
        self.teamRankings = []
        for i in range(len(self.ActiveTeamList)):
            self.teamRankings.append({'team':self.ActiveTeamList[i], 'score':0, 'wins':0, 'losses':0, 'draws':0})
        return self.teamRankings
    
    def update_team_rankings(self, match_results):
        self.match_results = match_results
        for i in range(len(self.match_results)):
            team1, team2, team1_score, team2_score, winner = self.match_results[i]
            for j in range(len(self.teamRankings)):
                if self.teamRankings[j]['team'] == team1:
                    if winner == team1:
                        self.teamRankings[j]['score'] += 3
                        self.teamRankings[j]['wins'] += 1
                    elif winner == team2:
                        self.teamRankings[j]['losses'] += 1
                    else:  # It's a draw
                        self.teamRankings[j]['score'] += 1
                        self.teamRankings[j]['draws'] += 1
                elif self.teamRankings[j]['team'] == team2:
                    if winner == team2:
                        self.teamRankings[j]['score'] += 3
                        self.teamRankings[j]['wins'] += 1
                    elif winner == team1:
                        self.teamRankings[j]['losses'] += 1
                    else:  # It's a draw
                        self.teamRankings[j]['score'] += 1
                        self.teamRankings[j]['draws'] += 1
        return self.teamRankings
    
    def get_match_results(self, match_ups):
        self.match_ups = match_ups
        self.match_results = []
        for i in range(len(self.match_ups)):
            team1_score = random.randint(0, 7)
            team2_score = random.randint(0, 7)
            if team1_score > team2_score:
                winner = self.match_ups[i][0]
            elif team1_score < team2_score:
                winner = self.match_ups[i][1]
            else:
                winner = "Draw"
            self.match_results.append([self.match_ups[i][0], self.match_ups[i][1], team1_score, team2_score, winner])
        return self.match_results
    def log_match_results(self, match_results):
        self.match_results = match_results
        for i in range(len(self.match_results)):
            print(self.match_results[i][0] + " " + str(self.match_results[i][2]) + " - " + str(self.match_results[i][3]) + " " + self.match_results[i][1])
            print("Winner: " + self.match_results[i][4])
            print(" ")

    def print_league_table(self, teamRankings):
        self.teamRankings = teamRankings
        print("Team\t\tScore\tWins\tLosses\tDraws")
        for i in range(len(self.teamRankings)):
            print(self.teamRankings[i]['team'] + "\t\t" + str(self.teamRankings[i]['score']) + "\t" + str(self.teamRankings[i]['wins']) + "\t" + str(self.teamRankings[i]['losses']) + "\t" + str(self.teamRankings[i]['draws']))


#Test implementation
testTournament = tournament(tName='testTournament', id=1, duration_days=5, prize_pool=1000, game_type='1v1')

teama = ['Eagles', 'Tigers', 'Lions', 'Bears', 'Sharks', 'Wolves']

testTournament.initialise_team_rankings(teama)
app_running = True
x = 0

while x < 5:
    testTournament.create_match_ups(teama)
    testTournament.get_match_results(testTournament.match_ups)
    testTournament.log_match_results(testTournament.match_results)
    testTournament.update_team_rankings(testTournament.match_results)
    print(testTournament.teamRankings)

    x = x +1

print('')
testTournament.print_league_table(testTournament.teamRankings)
x=0
print('')

while x < 5:
    testTournament.create_match_ups(teama)
    testTournament.get_match_results(testTournament.match_ups)
    testTournament.log_match_results(testTournament.match_results)
    testTournament.update_team_rankings(testTournament.match_results)
    print(testTournament.teamRankings)

    x = x +1

print('')
testTournament.print_league_table(testTournament.teamRankings)
print('')   