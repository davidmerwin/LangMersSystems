import random

class Game:
  def __init__(self):
    self.words = [("apple", "蘋果"), ("banana", "香蕉"), ("orange", "橙"), ("watermelon", "西瓜"), ("pineapple", "菠蘿")]
    self.currentWord = ("", "")
    self.attempts = 0
    self.score = 0

  def startNewGame(self):
    self.currentWord = random.choice(self.words)
    self.attempts = 0
    self.score = 0
    print("Guess the word in Cantonese!")
    print(self.currentWord[0])

  def checkGuess(self, guess):
    self.attempts += 1
    if guess.lower() == self.currentWord[1]:
      print(f"Congratulations! You guessed the word in {self.attempts} attempts.")
      self.score += 100 - self.attempts * 10
      self.startNewGame()
    else:
      print("Wrong! Try again.")

  def playGame(self):
    self.startNewGame()
    while True:
      guess = input("Enter your guess in Cantonese: ")
      self.checkGuess(guess)

game = Game()
game.playGame()
